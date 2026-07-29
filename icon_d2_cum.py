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
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xarray as xr

import earthkit.data
from earthkit.data import config

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.54

FILE_LAST_HOUR = "ultima_ora_icond2_cum.txt" 
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
    headers = {"User-Agent": "MeteoBot-ICOND2-Mappe/3.0"}
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
    thread_id = os.getenv("TELEGRAM_THREAD_ID_5")

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


def raggruppa_per_giorno(dt_run_utc: datetime) -> dict:
    rome_tz = pytz.timezone("Europe/Rome")
    giorni = {}
    for h in range(1, 49):
        start_utc = dt_run_utc + timedelta(hours=h-1)
        start_local = start_utc.astimezone(rome_tz)
        day_key = start_local.strftime("%Y-%m-%d")
        if day_key not in giorni:
            giorni[day_key] = []
        giorni[day_key].append(h)
    return giorni


def genera_album_cumulativo(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    giorni = raggruppa_per_giorno(dt_run_utc)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    domain = [xmin, xmax, ymin, ymax]

    # Nuova scala aggiornata
    my_levels = [0.5, 2, 5, 10, 20, 40, 60, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600]
    my_colors = [
        "#e6f2ff", "#99ccff", "#3399ff", "#004cff", "#66e666", "#33cc33", 
        "#009900", "#99cc00", "#ffe600", "#e6b300", "#ff9900", "#ff6600", 
        "#ff3300", "#ff3333", "#b30000", "#cc33ff", "#8000cc", "#4d0080"
    ]
    
    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    percorsi_foto = []
    cache_tot = {}

    for idx, (date_str, ore_list) in enumerate(giorni.items(), start=1):
        max_h = max(ore_list)

        print(f"\nGenerazione accumulo cumulativo totale fino al {date_str} (H+{max_h})...")

        try:
            if max_h not in cache_tot:
                cache_tot[max_h] = scarica_step_precipitazione(dt_run_utc, max_h)
            
            prec_cumulata = cache_tot[max_h]
            prec_mean_xr = prec_cumulata.mean(dim="eps")

            lat_vals = prec_mean_xr['latitude'].values
            lon_vals = prec_mean_xr['longitude'].values
            mean_vals = prec_mean_xr.values

            fig = plt.figure(figsize=(10, 8))
            ax = plt.axes(projection=ccrs.Mercator())
            ax.set_extent(domain, crs=ccrs.PlateCarree())

            ax.add_feature(regions_feature)
            if prov_feature: ax.add_feature(prov_feature)
            else: 
                ax.coastlines(resolution='10m')
                ax.add_feature(cfeature.BORDERS)

            cmap = ListedColormap(my_colors)
            norm = BoundaryNorm(my_levels, cmap.N)

            mask = mean_vals >= 0.5

            if np.any(mask):
                sc = ax.scatter(lon_vals[mask], lat_vals[mask], 
                                c=mean_vals[mask], cmap=cmap, norm=norm,
                                s=4, marker='s', transform=ccrs.PlateCarree(),
                                edgecolors='none')
                
                cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                cbar.set_label("Precipitazione Cumulata Totale Media (mm)", fontweight='bold')

            ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())
            for lo, la, sig in zip(lons, lats, sigle):
                ax.plot(lo, la, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                ax.text(lo + 0.05, la + 0.05, sig, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            start_local = dt_run_utc.astimezone(rome_tz)
            end_local = (dt_run_utc + timedelta(hours=max_h)).astimezone(rome_tz)
            str_valida = f"{start_local.strftime('%H:%M del %d/%m/%Y')} - {end_local.strftime('%H:%M del %d/%m/%Y')} (+{max_h}h)"

            title = f"ICON-D2 EPS - Precipitazione Cumulata Totale (mm)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}"
            plt.title(title, fontweight='bold')

            filename = f"accum_cum_{idx}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close(fig)
            percorsi_foto.append(filename)

        except Exception as e:
            print(f"  ❌ Errore elaborando la data {date_str}: {e}")
            continue

    if percorsi_foto:
        caption_album = f"ICON-D2 EPS: Precipitazione Cumulata Totale (mm)\nRun {nome_run} ({dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')})"
        invia_album_telegram(percorsi_foto, caption_album)
        
        for f in percorsi_foto:
            if os.path.exists(f): os.remove(f)

def main():
    print("Cerco l'ultimo run completo ICON-D2 EPS tramite la sentinella Open-Meteo...")
    data = fetch_dati_con_retry()
    
    if not data: 
        sys.exit(0)
        
    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)

    if is_new:
        print(f"🚀 Lancio generazione Precipitazione Cumulata Totale ICON-D2 per il RUN {nome_run} ({dt_run_utc.strftime('%Y-%m-%d %H:%M')})")
        genera_album_cumulativo(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run trovato o run in fase di caricamento. Uscita.")

if __name__ == "__main__":
    main()
