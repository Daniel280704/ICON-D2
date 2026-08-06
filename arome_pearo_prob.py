import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MPLBACKEND"] = "Agg"

import os, sys, time, json, requests, urllib3, pytz, tempfile, re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime, timedelta, timezone
import warnings
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import earthkit.data
from earthkit.data import config

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE, LONGITUDE = 45.07, 7.51

RUN_DURATION = 51
START_DELAY = 2

def fetch_dati_con_retry() -> dict:
    print("DEBUG: Interrogazione Open-Meteo per il controllo run...", flush=True)
    url_om = "https://api.open-meteo.com/v1/forecast"
    params_om = {
        "latitude": LATITUDE, 
        "longitude": LONGITUDE, 
        "hourly": "temperature_2m", 
        "models": "meteofrance_arome_seamless", 
        "timezone": "Europe/Rome", 
        "past_days": 1, 
        "forecast_days": 3
    }
    
    for tentativo in range(3):
        try:
            r = requests.get(url_om, params=params_om, headers={"User-Agent": "MeteoBot-AROME-Mappe/1.0"}, timeout=30)
            r.raise_for_status()
            print("DEBUG: Dati Open-Meteo scaricati con successo.", flush=True)
            return r.json()
        except Exception as e:
            print(f"DEBUG: Tentativo {tentativo+1} fallito su Open-Meteo: {e}", flush=True)
            time.sleep(15)
    return {}

def estrai_limiti_run(hourly_data: dict, ref_param: str, utc_offset_sec: int) -> tuple[bool, str, datetime]:
    times = hourly_data.get("time", [])
    mean_vals = hourly_data.get(ref_param, [])

    if not times or not mean_vals: return False, "", None

    end_idx = -1
    for i in range(len(mean_vals) - 1, -1, -1):
        if mean_vals[i] is not None:
            end_idx = i
            break

    if end_idx == -1: return False, "", None

    ultima_ora_valida_str = times[end_idx]

    dt_end_local = datetime.fromisoformat(ultima_ora_valida_str)
    dt_end_utc = dt_end_local - timedelta(seconds=utc_offset_sec)
    dt_run_utc_naive = dt_end_utc - timedelta(hours=RUN_DURATION)
    dt_start_utc = dt_run_utc_naive + timedelta(hours=START_DELAY)

    dt_start_local = dt_start_utc + timedelta(seconds=utc_offset_sec)
    start_time_str = dt_start_local.strftime("%Y-%m-%dT%H:%M")
    nome_run = dt_run_utc_naive.strftime("%H") + "Z"

    try:
        start_idx = times.index(start_time_str)
    except ValueError:
        return False, "", None

    expected_points = RUN_DURATION - START_DELAY + 1
    actual_points = end_idx - start_idx + 1

    if actual_points < expected_points:
        print(f"DEBUG: Run {nome_run} in caricamento... ({actual_points}/{expected_points} ore)", flush=True)
        return False, "", None

    print(f"DEBUG: Ultima ora trovata: {ultima_ora_valida_str} | Run corretto calcolato: {nome_run}", flush=True)
    print("DEBUG: Controllo cache disattivato per test. Procedo direttamente.", flush=True)

    return True, nome_run, dt_run_utc_naive.replace(tzinfo=timezone.utc)

def scopri_coverage_id_precipitazione(dt_run_utc, mf_token):
    print("DEBUG: Bypass GetCapabilities (Errore 500). Tento la scoperta diretta (Brute-force) del parametro...", flush=True)
    run_str_dot = dt_run_utc.strftime('%Y-%m-%dT%H.00.00Z')
    workspace = "MF-NWP-HIGHRES-PEARO000-0025-FRANCE-WCS"
    url_pearo = f"https://public-api.meteofrance.fr/public/pearome/1.0/wcs/{workspace}/GetCoverage"

    # Creiamo un micro-test di 1 pixel (velocissimo) per verificare quale parametro esiste
    dt_target_step = dt_run_utc + timedelta(hours=1)
    target_time_str = dt_target_step.strftime('%Y-%m-%dT%H:%M:%SZ')

    candidati = [
        f"TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE___{run_str_dot}_PT1H",
        f"TOTAL_WATER_PRECIPITATION__GROUND_OR_WATER_SURFACE___{run_str_dot}_PT1H",
        f"PRECIPITATION__GROUND_OR_WATER_SURFACE___{run_str_dot}_PT1H",
        f"TOTAL_PRECIPITATION__GROUND_OR_WATER_SURFACE___{run_str_dot}",
        f"TOTAL_WATER_PRECIPITATION__GROUND_OR_WATER_SURFACE___{run_str_dot}"
    ]

    for cov_id in candidati:
        print(f"DEBUG: Test ping su -> {cov_id}", flush=True)
        params_cov = [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageId", cov_id),
            ("format", "application/wmo-grib"),
            ("subset", f"time({target_time_str})"),
            ("subset", "lat(45.0,45.1)"),
            ("subset", "long(7.5,7.6)")
        ]
        
        try:
            r = requests.get(url_pearo, params=params_cov, headers={"apikey": mf_token}, timeout=15)
            if r.status_code == 200:
                print(f"DEBUG: BINGO! Parametro accettato: {cov_id}", flush=True)
                return cov_id
        except Exception:
            pass
        time.sleep(1)

    raise ValueError("Nessun parametro ha funzionato. Météo-France potrebbe avere i server down.")

def scarica_step_ensemble(dt_run_utc, h_step, cov_id_base, mf_token, max_retries=3):
    dt_target_step = dt_run_utc + timedelta(hours=h_step)
    target_time_str = dt_target_step.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    members_data = []
    print(f"DEBUG: Download Step {h_step} ({target_time_str}) - Membri estratti: ", end="", flush=True)

    for i in range(25):
        member_str = f"{i:03d}"
        workspace = f"MF-NWP-HIGHRES-PEARO{member_str}-0025-FRANCE-WCS"
        url_pearo = f"https://public-api.meteofrance.fr/public/pearome/1.0/wcs/{workspace}/GetCoverage"
        
        cov_id = cov_id_base.replace("000", member_str)

        params_cov = [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("coverageId", cov_id),
            ("format", "application/wmo-grib"),
            ("subset", f"time({target_time_str})"),
            ("subset", "lat(43.0,47.0)"),
            ("subset", "long(5.5,11.0)")
        ]

        for tentativo in range(max_retries):
            try:
                r = requests.get(url_pearo, params=params_cov, headers={"apikey": mf_token}, stream=True, timeout=30)
                r.raise_for_status()
                
                fd, temp_path = tempfile.mkstemp(suffix=".grib2")
                with os.fdopen(fd, 'wb') as f_out:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: 
                            f_out.write(chunk)
                        
                ds = earthkit.data.from_source("file", temp_path).to_xarray()
                var_name = list(ds.data_vars)[0]
                da = ds[var_name].compute()
                ds.close()
                
                try: 
                    os.remove(temp_path)
                except: 
                    pass
                
                members_data.append(da)
                print(f"[{member_str}]", end="", flush=True)
                break
                
            except Exception as e:
                if tentativo == max_retries - 1:
                    print(f"[X_{member_str}]", end="", flush=True)
                    raise e
                time.sleep(2)

    print("", flush=True) 
    return xr.concat(members_data, dim="eps")

def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_3")
    
    if not token or not chat_id: 
        return
        
    if len(file_paths) == 1:
        try:
            with open(file_paths[0], "rb") as f: 
                payload = {"chat_id": chat_id, "caption": caption, "message_thread_id": thread_id}
                requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", data=payload, files={"photo": f})
        except: 
            pass
        return
        
    media = []
    for i in range(len(file_paths)):
        item = {
            "type": "photo",
            "media": f"attach://photo_{i}",
            "caption": caption if i == 0 else ""
        }
        media.append(item)
        
    files = {}
    for i, p in enumerate(file_paths):
        files[f"photo_{i}"] = open(p, "rb")
        
    try: 
        payload = {"chat_id": chat_id, "media": json.dumps(media), "message_thread_id": thread_id}
        requests.post(f"https://api.telegram.org/bot{token}/sendMediaGroup", data=payload, files=files)
    except: 
        pass
    finally:
        for f in files.values(): 
            f.close()

def raggruppa_in_blocchi(dt_run_local: datetime) -> dict:
    blocchi = {}
    for h in range(1, 49): 
        dt_target = dt_run_local + timedelta(hours=h)
        date_str = dt_target.strftime("%Y-%m-%d")
        hour = dt_target.hour
        
        if hour == 0 or 18 <= hour <= 23: 
            b_name = "18-24"
            if hour == 0:
                date_str = (dt_target - timedelta(days=1)).strftime("%Y-%m-%d")
        elif 1 <= hour <= 6: 
            b_name = "00-06"
        elif 7 <= hour <= 12: 
            b_name = "06-12"
        else: 
            b_name = "12-18"
            
        key = f"{date_str} (Fascia {b_name})"
        if key not in blocchi: 
            blocchi[key] = []
        blocchi[key].append(h)
        
    return blocchi

def genera_album_orari(dt_run_utc: datetime, nome_run: str):
    mf_token = os.getenv("METEO_FRANCE_TOKEN")
    if not mf_token:
        print("DEBUG: Manca il token di Météo-France.", flush=True)
        return
        
    try:
        cov_id_base = scopri_coverage_id_precipitazione(dt_run_utc, mf_token)
    except Exception as e:
        print(f"DEBUG: Impossibile ricavare il parametro precipitazione: {e}", flush=True)
        return

    # Se il parametro ha il suffisso _PT1H, i dati sono già su base oraria nativa
    is_hourly_native = "_PT1H" in cov_id_base
    if is_hourly_native:
        print("DEBUG: WOW! Il server fornisce gli accumuli orari nativi (_PT1H). Nessuna sottrazione necessaria!", flush=True)

    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    blocchi = raggruppa_in_blocchi(dt_run_local)
    
    domain = [6.0, 10.5, 43.5, 46.8]
    my_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    my_colors = ["#a0e6ff", "#00a0ff", "#00ff00", "#ffff00", "#ffaa00", "#ff0000", "#cc0000", "#ff00ff", "#800080"]
    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        valid_geoms = [geom for geom in shpreader.Reader(shp_path).geometries() if not geom.is_empty]
        prov_feature = cfeature.ShapelyFeature(valid_geoms, ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')
    else:
        prov_feature = None

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68, 7.55, 8.20, 8.61, 8.42, 8.61, 8.05, 8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione probabilità orarie PEARO: {block_name}", flush=True)
        percorsi_foto = []
        prev_step_idx = -1
        prev_tot = None
        
        for h in ore_list:
            try:
                curr_tot = scarica_step_ensemble(dt_run_utc, h, cov_id_base, mf_token)
                
                # Sfruttiamo l'orario nativo per saltare il download e la sottrazione del passo precedente
                if is_hourly_native or h == 1: 
                    prec_oraria = curr_tot
                else:
                    if prev_step_idx == h - 1:
                        prec_h_minus_1 = prev_tot
                    else:
                        prec_h_minus_1 = scarica_step_ensemble(dt_run_utc, h - 1, cov_id_base, mf_token)
                    
                    prec_oraria = curr_tot.copy(data=np.maximum(0, curr_tot.values - prec_h_minus_1.values))
                
                prev_tot = curr_tot
                prev_step_idx = h

                mean_xr = (prec_oraria >= 0.5).astype(float).mean(dim="eps") * 100
                lat_vals = mean_xr['latitude'].values.flatten()
                lon_vals = mean_xr['longitude'].values.flatten()
                mean_vals = mean_xr.values.flatten()
                
                mask_nw = (lat_vals >= 43.5) & (lat_vals <= 46.8) & (lon_vals >= 6.0) & (lon_vals <= 10.5)
                lat_crop = lat_vals[mask_nw]
                lon_crop = lon_vals[mask_nw]
                mean_crop = np.nan_to_num(mean_vals[mask_nw], nan=0.0)

                fig = plt.figure(figsize=(10, 8))
                ax = plt.axes(projection=ccrs.Mercator())
                ax.set_extent(domain, crs=ccrs.PlateCarree())
                ax.add_feature(regions_feature)
                
                if prov_feature: 
                    ax.add_feature(prov_feature)
                else: 
                    ax.coastlines(resolution='10m')
                    ax.add_feature(cfeature.BORDERS)

                cmap = ListedColormap(my_colors)
                norm = BoundaryNorm(my_levels, len(my_colors))
                
                if np.max(mean_crop) >= my_levels[0]:
                    cf = ax.tricontourf(lon_crop, lat_crop, mean_crop, levels=my_levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=1.0)
                    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                    cbar.set_label("Probabilità (%)", fontweight='bold')
                else:
                    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([]) 
                    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                    cbar.set_label("Probabilità (%)", fontweight='bold')

                ax.plot(LONGITUDE, LATITUDE, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())
                for lo, la, sig in zip(lons, lats, sigle):
                    ax.plot(lo, la, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                    ax.text(lo + 0.05, la + 0.05, sig, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

                start_local = dt_run_local + timedelta(hours=h-1)
                end_local = dt_run_local + timedelta(hours=h)
                
                title = f"PEARO (AROME EPS) - Probabilità Pioggia >= 0.5 mm/h (%)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M del %d/%m')}"
                plt.title(title, fontweight='bold')
                
                filename = f"oraria_prob_{h}.png"
                plt.savefig(filename, dpi=200, bbox_inches='tight')
                plt.close(fig)
                percorsi_foto.append(filename)
                
            except Exception as e:
                print(f"  ⚠️ [SKIP] Errore ora {h}: {e}", flush=True)
                continue

        if percorsi_foto:
            invia_album_telegram(percorsi_foto, f"PEARO EPS: Probabilità Pioggia oraria >= 0.5 mm\n{block_name}\nRun {nome_run}")
            for f in percorsi_foto:
                if os.path.exists(f): 
                    os.remove(f)

if __name__ == "__main__":
    print("DEBUG: Avvio script PEARO...", flush=True)
    data = fetch_dati_con_retry()
    
    if data:
        is_new, nome_run, dt_run_utc = estrai_limiti_run(data.get("hourly", {}), "temperature_2m", data.get("utc_offset_seconds", 0))
        if is_new: 
            genera_album_orari(dt_run_utc, nome_run)
    else:
        print("DEBUG: Impossibile recuperare i dati da Open-Meteo.", flush=True)
