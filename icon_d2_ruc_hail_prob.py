import os
import sys
import time
import json
import requests
import urllib3
import pytz
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime, timedelta, timezone
from scipy.interpolate import griddata
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

FILE_LAST_HOUR = "ultima_ora_icond2_ruc_hail_prob.txt" 
RUN_DURATION = 27 

def trova_ultimo_run_completo(session: requests.Session) -> tuple[bool, datetime, str]:
    now = datetime.now(timezone.utc)
    for i in range(6):
        dt_run = now - timedelta(hours=i)
        run_str = dt_run.strftime("%Y-%m-%dT%H:00")
        test_url = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc-eps/p/KEF_HAIL_MAX_S/r/{run_str}/e/20/s/PT027H00M.grib2"
        
        try:
            resp = session.head(test_url, timeout=10)
            if resp.status_code == 200:
                if os.path.exists(FILE_LAST_HOUR):
                    with open(FILE_LAST_HOUR, "r") as f:
                        if run_str <= f.read().strip():
                            print(f"✅ Run ICON-D2-RUC-EPS Probabilità {run_str} già elaborato.")
                            return False, None, None
                
                with open(FILE_LAST_HOUR, "w") as f:
                    f.write(run_str)
                return True, dt_run, run_str
        except Exception:
            continue
            
    return False, None, None


def scarica_step_grib(session: requests.Session, run_str: str, member: int, h: int, m: int, max_retries=2):
    url = f"https://opendata.dwd.de/weather/nwp/v1/m/icon-d2-ruc-eps/p/KEF_HAIL_MAX_S/r/{run_str}/e/{member:02d}/s/PT{h:03d}H{m:02d}M.grib2"
    
    for tentativo in range(max_retries):
        try:
            r = session.get(url, stream=True, timeout=15)
            r.raise_for_status()
            
            fd, temp_path = tempfile.mkstemp(suffix=".grib2")
            with os.fdopen(fd, 'wb') as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: f_out.write(chunk)
            
            ds = earthkit.data.from_source("file", temp_path).to_xarray()
            var_name = list(ds.data_vars)[0]
            
            raw_data = ds[var_name].values
            lats = ds['latitude'].values
            lons = ds['longitude'].values
            
            ds.close()
            os.remove(temp_path)
            
            # Conversione in centimetri
            if np.nanmax(raw_data) < 1.0: 
                data_cm = raw_data * 100.0
            elif np.nanmax(raw_data) > 50:
                data_cm = raw_data / 10.0
            else:
                data_cm = raw_data

            return data_cm, (lats, lons)

        except Exception as e:
            if tentativo == max_retries - 1: return None, None
            time.sleep(2)
            
    return None, None


def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_121")

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
    media, files = [], {}

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
        print(f"📸 Album Telegram PROBABILITÀ GRANDINE (RUC) inviato ({len(file_paths)} mappe).")
    except Exception as e:
        print(f"Errore invio album Telegram: {e}")
    finally:
        for f in files.values():
            f.close()


def raggruppa_in_blocchi(dt_run_local: datetime) -> dict:
    blocchi = {}
    for h in range(1, RUN_DURATION + 1): 
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


def genera_album_probabilita_grandine(session: requests.Session, dt_run_utc: datetime, run_str: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    domain = [xmin, xmax, ymin, ymax]

    # Griglia regolare per il rendering a chiazze
    grid_lon = np.linspace(xmin, xmax, 300)
    grid_lat = np.linspace(ymin, ymax, 300)
    grid_lon2d, grid_lat2d = np.meshgrid(grid_lon, grid_lat)

    # Palette e livelli per la probabilità
    my_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    my_colors = ["#a0e6ff", "#00a0ff", "#00ff00", "#ffff00", "#ffaa00", "#ff0000", "#cc0000", "#ff00ff", "#800080"]

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats_plot = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons_plot = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    lats_grid, lons_grid = None, None

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione album Probabilità Grandine ICON-D2-RUC: {block_name}")
        percorsi_foto = []

        for h in ore_list:
            print(f"  ⬇️  Elaborazione H={h} (Calcolo probabilità membri >= 0.5 cm)...")
            
            exceed_count = None
            valid_members_count = 0
            
            for member in range(1, 21):
                member_max = None
                steps = [(h - 1, 15), (h - 1, 30), (h - 1, 45), (h, 0)]
                
                for step_h, step_m in steps:
                    data_cm, coords = scarica_step_grib(session, run_str, member, step_h, step_m)
                    
                    if data_cm is not None:
                        if lats_grid is None: lats_grid, lons_grid = coords
                        if member_max is None: member_max = data_cm
                        else: member_max = np.maximum(member_max, data_cm)
                            
                if member_max is not None:
                    # Verifichiamo dove il membro individua grandine >= 0.5 cm
                    is_exceeded = (member_max >= 0.5).astype(float)
                    if exceed_count is None: exceed_count = is_exceeded
                    else: exceed_count += is_exceeded
                    valid_members_count += 1
            
            if exceed_count is not None and valid_members_count > 0:
                # Calcolo della percentuale di membri favorevoli
                prob_vals = (exceed_count / valid_members_count) * 100.0
                
                # Interpolazione a chiazze
                pts = np.column_stack((lons_grid.ravel(), lats_grid.ravel()))
                vals = prob_vals.ravel()
                
                valid_mask = ~np.isnan(vals) & (vals >= 5.0)
                
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

                if np.any(valid_mask):
                    grid_val = griddata(pts[valid_mask], vals[valid_mask], (grid_lon2d, grid_lat2d), method='linear', fill_value=np.nan)
                    
                    # Mascheriamo le probabilità inferiori al 10%
                    grid_val_masked = np.ma.masked_where(np.isnan(grid_val) | (grid_val < 10.0), grid_val)

                    sc = ax.pcolormesh(grid_lon2d, grid_lat2d, grid_val_masked,
                                       cmap=cmap, norm=norm,
                                       transform=ccrs.PlateCarree(),
                                       shading='auto')
                    
                    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                    cbar.set_label("Probabilità Grandine >= 0.5 cm (%) - RUC EPS", fontweight='bold')

                ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())
                for lo, la, sig in zip(lons_plot, lats_plot, sigle):
                    ax.plot(lo, la, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                    ax.text(lo + 0.05, la + 0.05, sig, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

                dt_target_local = (dt_run_utc + timedelta(hours=h)).astimezone(rome_tz)
                str_valida = f"Ora {dt_target_local.strftime('%H:%M del %d/%m/%Y')} (+{h}h)"

                title = f"ICON-D2-RUC EPS - Probabilità Grandine >= 0.5 cm (%)\nRun: {run_str} UTC | Target: {str_valida}"
                plt.title(title, fontweight='bold')

                filename = f"ruc_hail_prob_{h}.png"
                plt.savefig(filename, dpi=200, bbox_inches='tight')
                plt.close(fig)
                percorsi_foto.append(filename)

        if percorsi_foto:
            nome_run = run_str.split('T')[-1].replace(':00', 'Z')
            caption_album = f"ICON-D2-RUC EPS: Probabilità Grandine (>= 0.5 cm)\n{block_name}\nRun {nome_run}"
            invia_album_telegram(percorsi_foto, caption_album)
            
            for f in percorsi_foto:
                if os.path.exists(f): os.remove(f)
                
        time.sleep(10)

def main():
    print("Ricerca dell'ultimo run ICON-D2-RUC EPS (Probabilità Grandine)...")
    with requests.Session() as session:
        session.headers.update({"User-Agent": "MeteoBot-ICOND2-RUC/3.0"})
        is_new, dt_run_utc, run_str = trova_ultimo_run_completo(session)

        if is_new:
            print(f"🚀 Lancio generazione Probabilità Grandine ICON-D2-RUC EPS per il RUN {run_str}")
            genera_album_probabilita_grandine(session, dt_run_utc, run_str)
        else:
            print("Nessun nuovo run trovato. Uscita.")

if __name__ == "__main__":
    main()