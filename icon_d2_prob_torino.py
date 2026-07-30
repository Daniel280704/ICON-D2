import os
import sys
import time
import json
import requests
import urllib3
import pytz
import bz2
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib.cm as cm
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import cartopy.io.img_tiles as cimgt # Modulo Google Tiles aggiunto
import xarray as xr

import earthkit.data
from earthkit.data import config

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.68

FILE_LAST_HOUR = "ultima_ora_icond2_prob_to.txt" 
RUN_DURATION = 48 
START_DELAY = 0

def fetch_dati_con_retry() -> dict:
    URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m", 
        "models": "dwd_icon_d2_eps_ensemble_mean",
        "timezone": "Europe/Rome",
        "past_days": 1,
        "forecast_days": 3 
    }
    headers = {"User-Agent": "MeteoBot-ICOND2-Mappe-TO/3.0"}
    for tentativo in range(3):
        try:
            r = requests.get(URL, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"⚠️ Errore API Open-Meteo: {e}")
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
        print(f"⏳ Run {nome_run} in caricamento su Open-Meteo... ({actual_points}/{expected_points} ore)")
        return False, "", None

    if os.path.exists(FILE_LAST_HOUR):
        with open(FILE_LAST_HOUR, "r") as f:
            ultima_ora_salvata = f.read().strip()
        if ultima_ora_valida_str <= ultima_ora_salvata:
            print(f"✅ Run ICON-D2 EPS {nome_run} già elaborato (Ultimo blocco: {ultima_ora_valida_str}).")
            return False, "", None

    with open(FILE_LAST_HOUR, "w") as f:
        f.write(ultima_ora_valida_str)

    dt_run_utc = dt_run_utc_naive.replace(tzinfo=timezone.utc)
    return True, nome_run, dt_run_utc


def scarica_step_precipitazione(dt_run_utc, h_step, max_retries=3):
    run_hour_syn = dt_run_utc.hour          
    run_hour = f"{run_hour_syn:02d}"
    date_hour = dt_run_utc.strftime('%Y%m%d%H')
    step_str = f"{h_step:03d}"
    
    url_gsp = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/rain_gsp/icon-d2-eps_germany_icosahedral_single-level_{date_hour}_{step_str}_2d_rain_gsp.grib2.bz2"
    url_con = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/rain_con/icon-d2-eps_germany_icosahedral_single-level_{date_hour}_{step_str}_2d_rain_con.grib2.bz2"

    def _download_one(url: str):
        for tentativo in range(max_retries):
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                fd, temp_path = tempfile.mkstemp(suffix=".grib2")
                with os.fdopen(fd, 'wb') as f_out:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: f_out.write(decompressor.decompress(chunk))
                return temp_path
            except Exception as e:
                if tentativo == max_retries - 1: raise e
                time.sleep(5 * (tentativo + 1))

    p_gsp = _download_one(url_gsp)
    p_con = _download_one(url_con)

    ds_gsp = earthkit.data.from_source("file", p_gsp).to_xarray()
    ds_con = earthkit.data.from_source("file", p_con).to_xarray()

    if 'member' in ds_gsp.dims: ds_gsp = ds_gsp.rename({'member': 'eps'})
    elif 'number' in ds_gsp.dims: ds_gsp = ds_gsp.rename({'number': 'eps'})
    
    if 'member' in ds_con.dims: ds_con = ds_con.rename({'member': 'eps'})
    elif 'number' in ds_con.dims: ds_con = ds_con.rename({'number': 'eps'})

    gsp_var = list(ds_gsp.data_vars)[0]
    con_var = list(ds_con.data_vars)[0]
    
    tot_prec = (ds_gsp[gsp_var] + ds_con[con_var]).compute()

    ds_gsp.close()
    ds_con.close()
    try: os.remove(p_gsp) 
    except: pass
    try: os.remove(p_con) 
    except: pass

    return tot_prec


def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_3873")

    if not token or not chat_id: return

    if len(file_paths) == 1:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        payload = {"chat_id": chat_id, "caption": caption}
        if thread_id: payload["message_thread_id"] = thread_id
        try:
            with open(file_paths[0], "rb") as photo:
                requests.post(url, data=payload, files={"photo": photo})
        except Exception as e:
            print(f"Errore invio singola foto: {e}")
        return

    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    media = []
    files = {}

    for idx, path in enumerate(file_paths):
        media.append({
            "type": "photo",
            "media": f"attach://photo_{idx}",
            "caption": caption if idx == 0 else ""
        })
        files[f"photo_{idx}"] = open(path, "rb")

    payload = {"chat_id": chat_id, "media": json.dumps(media)}
    if thread_id: payload["message_thread_id"] = thread_id

    try:
        requests.post(url, data=payload, files=files)
        print(f"📸 Album Telegram inviato con successo ({len(file_paths)} mappe).")
    except Exception as e:
        print(f"Errore invio album Telegram: {e}")
    finally:
        for f in files.values():
            f.close()


def raggruppa_in_blocchi(dt_run_local: datetime) -> dict:
    blocchi = {}
    for h in range(1, 49): 
        dt_target = dt_run_local + timedelta(hours=h)
        date_str = dt_target.date().strftime("%Y-%m-%d")
        hour = dt_target.hour

        if hour == 0:
            date_str = (dt_target.date() - timedelta(days=1)).strftime("%Y-%m-%d")
            b_name = "18-24"
        elif 1 <= hour <= 6: b_name = "00-06"
        elif 7 <= hour <= 12: b_name = "06-12"
        elif 13 <= hour <= 18: b_name = "12-18"
        else: b_name = "18-24"

        key = f"{date_str} (Fascia {b_name})"
        if key not in blocchi:
            blocchi[key] = []
        blocchi[key].append(h)
    return blocchi


def genera_album_orari(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.6, 8.2, 44.7, 45.6
    my_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    my_colors = ["#a0e6ff", "#00a0ff", "#00ff00", "#ffff00", "#ffaa00", "#ff0000", "#cc0000", "#ff00ff", "#800080"]
    domain = [xmin, xmax, ymin, ymax]
    
    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    cities = {
        "Torino": (7.686, 45.070, "red"),
        "Rivoli": (7.516, 45.069, "red"),
        "Ivrea": (7.876, 45.467, "black"),
        "Cuorgnè": (7.650, 45.390, "black"),
        "Lanzo": (7.481, 45.272, "black"),
        "Rivarolo": (7.721, 45.331, "black"),
        "Ciriè": (7.602, 45.234, "black"),
        "Venaria": (7.628, 45.122, "black"),
        "Volpiano": (7.778, 45.201, "black"),
        "Chivasso": (7.888, 45.190, "black"),
        "Settimo": (7.766, 45.138, "black"),
        "Avigliana": (7.397, 45.079, "black"),
        "Chieri": (7.822, 45.011, "black"),
        "Moncalieri": (7.682, 45.000, "black"),
        "Carignano": (7.674, 44.906, "black"),
        "Pinerolo": (7.333, 44.883, "black"),
        "Susa": (7.045, 45.141, "black"),
        "Carmagnola": (7.716, 44.848, "black")
    }

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione album probabilità: {block_name}")
        percorsi_foto = []

        prev_step_idx = -1
        prev_tot = None

        for h in ore_list:
            try:
                print(f"  ⬇️  Elaborazione accumulo orario H={h}...")
                curr_tot = scarica_step_precipitazione(dt_run_utc, h)

                if h == 1:
                    prec_oraria = curr_tot
                else:
                    if prev_step_idx == h - 1 and prev_tot is not None:
                        prec_h_minus_1 = prev_tot
                    else:
                        prec_h_minus_1 = scarica_step_precipitazione(dt_run_utc, h - 1)
                    prec_oraria = curr_tot - prec_h_minus_1

                prev_tot = curr_tot
                prev_step_idx = h

                prob_xr = (prec_oraria >= 0.5).astype(float).mean(dim="eps") * 100
                
                lat_vals = prob_xr['latitude'].values
                lon_vals = prob_xr['longitude'].values
                prob_vals = prob_xr.values
                
                margin = 0.1
                domain_mask = (lon_vals >= xmin - margin) & (lon_vals <= xmax + margin) & \
                              (lat_vals >= ymin - margin) & (lat_vals <= ymax + margin)
                
                lon_crop = lon_vals[domain_mask]
                lat_crop = lat_vals[domain_mask]
                prob_crop = prob_vals[domain_mask]

                fig = plt.figure(figsize=(10, 8))
                ax = plt.axes(projection=ccrs.Mercator())
                ax.set_extent(domain, crs=ccrs.PlateCarree())

                # --- IMPLEMENTAZIONE GOOGLE MAPS HD ---
                # lyrs=p : Terrain/Physical + Strade/Autostrade + Fiumi
                tiler = cimgt.GoogleTiles(url="https://mt0.google.com/vt/lyrs=p&hl=it&x={x}&y={y}&z={z}")
                ax.add_image(tiler, 11) # Zoom livello 11 (dettaglio eccellente per scala provinciale)
                
                ax.add_feature(regions_feature)
                if prov_feature: ax.add_feature(prov_feature)
                else: 
                    ax.coastlines(resolution='10m')
                    ax.add_feature(cfeature.BORDERS)

                cmap = ListedColormap(my_colors)
                norm = BoundaryNorm(my_levels, cmap.N)

                if np.max(prob_crop) >= 10:
                    cf = ax.tricontourf(lon_crop, lat_crop, prob_crop, 
                                        levels=my_levels, cmap=cmap, norm=norm,
                                        transform=ccrs.PlateCarree(), alpha=0.7)
                    
                # --- LEGENDA SEMPRE VISIBILE ---
                sm = cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                cbar.set_label("Probabilità (%)", fontweight='bold')

                for name, (lo, la, col) in cities.items():
                    ax.plot(lo, la, marker='o', color=col, markersize=5 if col=='black' else 7, transform=ccrs.PlateCarree())
                    ax.text(lo + 0.015, la + 0.015, name, color=col, fontsize=8, fontweight='bold', transform=ccrs.PlateCarree(), path_effects=[PathEffects.withStroke(linewidth=2, foreground='white')])

                start_local = dt_run_local + timedelta(hours=h-1)
                end_local = dt_run_local + timedelta(hours=h)
                str_valida = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M del %d/%m')}"

                title = f"ICON-D2 EPS - Probabilità Pioggia >= 0.5 mm/h (%)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}"
                plt.title(title, fontweight='bold')

                filename = f"oraria_to_{h}.png"
                plt.savefig(filename, dpi=200, bbox_inches='tight')
                plt.close(fig)
                percorsi_foto.append(filename)

            except Exception as e:
                print(f"  ❌ Errore elaborando l'ora {h}: {e}")
                continue

        if percorsi_foto:
            caption_album = f"ICON-D2 EPS: Probabilità Pioggia oraria >= 0.5 mm (Torinese)\n{block_name}\nRun {nome_run}"
            invia_album_telegram(percorsi_foto, caption_album)
            
            for f in percorsi_foto:
                if os.path.exists(f): os.remove(f)
                
        time.sleep(10)

def main():
    print("Cerco l'ultimo run completo ICON-D2 EPS tramite la sentinella Open-Meteo...")
    data = fetch_dati_con_retry()
    
    if not data: 
        sys.exit(0)
        
    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)

    if is_new:
        print(f"🚀 Lancio generazione Probabilità Orarie ICON-D2 per il RUN {nome_run} ({dt_run_utc.strftime('%Y-%m-%d %H:%M')})")
        genera_album_orari(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run trovato o run in fase di caricamento. Uscita.")

if __name__ == '__main__':
    main()
