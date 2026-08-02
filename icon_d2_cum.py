import os, sys, time, json, requests, urllib3, pytz, bz2, tempfile
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

warnings.filterwarnings('ignore'); urllib3.disable_warnings(); config.set("cache-policy", "temporary")
LATITUDE, LONGITUDE = 45.07, 7.54
FILE_LAST_HOUR, RUN_DURATION, START_DELAY = "ultima_ora_icond2_cum.txt", 48, 0

def fetch_dati_con_retry() -> dict:
    URL, params = "https://ensemble-api.open-meteo.com/v1/ensemble", {"latitude": LATITUDE, "longitude": LONGITUDE, "hourly": "temperature_2m", "models": "dwd_icon_d2_eps_ensemble_mean", "timezone": "Europe/Rome", "past_days": 1, "forecast_days": 3}
    for _ in range(3):
        try:
            r = requests.get(URL, params=params, headers={"User-Agent": "MeteoBot-ICOND2-Mappe/3.0"}, timeout=30)
            r.raise_for_status(); return r.json()
        except: time.sleep(15)
    return {}

def estrai_limiti_run(hourly_data: dict, ref_param: str, utc_offset_sec: int) -> tuple[bool, str, datetime]:
    times, mean_vals = hourly_data.get("time", []), hourly_data.get(ref_param, [])
    if not times or not mean_vals: return False, "", None
    end_idx = next((i for i in range(len(mean_vals)-1, -1, -1) if mean_vals[i] is not None), -1)
    if end_idx == -1: return False, "", None
    ultima_ora_valida_str = times[end_idx]
    dt_end_local = datetime.fromisoformat(ultima_ora_valida_str)
    dt_run_utc_naive = dt_end_local - timedelta(seconds=utc_offset_sec) - timedelta(hours=RUN_DURATION)
    dt_start_local = dt_run_utc_naive + timedelta(hours=START_DELAY) + timedelta(seconds=utc_offset_sec)
    nome_run = dt_run_utc_naive.strftime("%H") + "Z"
    try: start_idx = times.index(dt_start_local.strftime("%Y-%m-%dT%H:%M"))
    except ValueError: return False, "", None
    if (end_idx - start_idx + 1) < (RUN_DURATION - START_DELAY + 1): return False, "", None
    if os.path.exists(FILE_LAST_HOUR):
        with open(FILE_LAST_HOUR, "r") as f:
            if ultima_ora_valida_str <= f.read().strip(): return False, "", None
    with open(FILE_LAST_HOUR, "w") as f: f.write(ultima_ora_valida_str)
    return True, nome_run, dt_run_utc_naive.replace(tzinfo=timezone.utc)

def scarica_step_precipitazione(dt_run_utc, h_step, max_retries=3):
    run_hour, date_hour, step_str = f"{dt_run_utc.hour:02d}", dt_run_utc.strftime('%Y%m%d%H'), f"{h_step:03d}"
    url_gsp = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/rain_gsp/icon-d2-eps_germany_icosahedral_single-level_{date_hour}_{step_str}_2d_rain_gsp.grib2.bz2"
    url_con = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/rain_con/icon-d2-eps_germany_icosahedral_single-level_{date_hour}_{step_str}_2d_rain_con.grib2.bz2"
    def _download_one(url: str):
        for tentativo in range(max_retries):
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status(); fd, temp_path = tempfile.mkstemp(suffix=".grib2")
                with os.fdopen(fd, 'wb') as f_out:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: f_out.write(decompressor.decompress(chunk))
                return temp_path
            except Exception as e:
                if tentativo == max_retries - 1: raise e
                time.sleep(5)
    p_gsp, p_con = _download_one(url_gsp), _download_one(url_con)
    ds_gsp, ds_con = earthkit.data.from_source("file", p_gsp).to_xarray(), earthkit.data.from_source("file", p_con).to_xarray()
    if 'member' in ds_gsp.dims: ds_gsp = ds_gsp.rename({'member': 'eps'})
    elif 'number' in ds_gsp.dims: ds_gsp = ds_gsp.rename({'number': 'eps'})
    if 'member' in ds_con.dims: ds_con = ds_con.rename({'member': 'eps'})
    elif 'number' in ds_con.dims: ds_con = ds_con.rename({'number': 'eps'})
    tot_prec = (ds_gsp[list(ds_gsp.data_vars)[0]] + ds_con[list(ds_con.data_vars)[0]]).compute()
    ds_gsp.close(); ds_con.close()
    try: os.remove(p_gsp); os.remove(p_con)
    except: pass
    return tot_prec

def invia_album_telegram(file_paths: list, caption: str):
    token, chat_id, thread_id = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("TELEGRAM_THREAD_ID_5")
    if not token or not chat_id: return
    if len(file_paths) == 1:
        try:
            with open(file_paths[0], "rb") as f: requests.post(f"https://api.telegram.org/bot{token}/sendPhoto", data={"chat_id": chat_id, "caption": caption, "message_thread_id": thread_id}, files={"photo": f})
        except: pass
        return
    media = [{"type": "photo", "media": f"attach://photo_{i}", "caption": caption if i == 0 else ""} for i in range(len(file_paths))]
    files = {f"photo_{i}": open(p, "rb") for i, p in enumerate(file_paths)}
    try: requests.post(f"https://api.telegram.org/bot{token}/sendMediaGroup", data={"chat_id": chat_id, "media": json.dumps(media), "message_thread_id": thread_id}, files=files)
    except: pass
    finally:
        for f in files.values(): f.close()

def raggruppa_per_giorno(dt_run_utc: datetime) -> dict:
    rome_tz, giorni = pytz.timezone("Europe/Rome"), {}
    for h in range(1, 49):
        day_key = (dt_run_utc + timedelta(hours=h-1)).astimezone(rome_tz).strftime("%Y-%m-%d")
        if day_key not in giorni: giorni[day_key] = []
        giorni[day_key].append(h)
    return giorni

def genera_album_cumulativo(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    giorni = raggruppa_per_giorno(dt_run_utc)
    domain = [6.0, 10.5, 43.5, 46.8]
    my_levels = [0.5, 2, 5, 10, 20, 40, 60, 80, 100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600]
    my_colors = ["#e6f2ff", "#99ccff", "#3399ff", "#004cff", "#66e666", "#33cc33", "#009900", "#99cc00", "#ffe600", "#e6b300", "#ff9900", "#ff6600", "#ff3300", "#ff3333", "#b30000", "#cc33ff", "#8000cc", "#4d0080"]
    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':') if os.path.exists(shp_path) else None
    lats, lons, sigle = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92], [7.68, 7.55, 8.20, 8.61, 8.42, 8.61, 8.05, 8.55], ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]
    percorsi_foto, cache_tot = [], {}

    for idx, (date_str, ore_list) in enumerate(giorni.items(), start=1):
        max_h = max(ore_list)
        print(f"\nGenerazione accumulo cumulativo totale per {date_str} (H+{max_h})...")
        try:
            if max_h not in cache_tot: cache_tot[max_h] = scarica_step_precipitazione(dt_run_utc, max_h)
            mean_xr = cache_tot[max_h].mean(dim="eps")

            lat_vals, lon_vals, mean_vals = mean_xr['latitude'].values.flatten(), mean_xr['longitude'].values.flatten(), mean_xr.values.flatten()
            mask_nw = (lat_vals >= 43.5) & (lat_vals <= 46.8) & (lon_vals >= 6.0) & (lon_vals <= 10.5)
            lat_crop, lon_crop, mean_crop = lat_vals[mask_nw], lon_vals[mask_nw], np.nan_to_num(mean_vals[mask_nw], nan=0.0)

            fig = plt.figure(figsize=(10, 8)); ax = plt.axes(projection=ccrs.Mercator()); ax.set_extent(domain, crs=ccrs.PlateCarree())
            ax.add_feature(regions_feature)
            if prov_feature: ax.add_feature(prov_feature)
            else: ax.coastlines(resolution='10m'); ax.add_feature(cfeature.BORDERS)

            cmap, norm = ListedColormap(my_colors), BoundaryNorm(my_levels, len(my_colors))
            if np.max(mean_crop) >= my_levels[0]:
                cf = ax.tricontourf(lon_crop, lat_crop, mean_crop, levels=my_levels, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), extend='max', alpha=1.0)
                cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05); cbar.set_label("Precipitazione Cumulata Totale Media (mm)", fontweight='bold')
            else:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([]) 
                cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05); cbar.set_label("Precipitazione Cumulata Totale Media (mm)", fontweight='bold')

            ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())
            for lo, la, sig in zip(lons, lats, sigle):
                ax.plot(lo, la, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                ax.text(lo + 0.05, la + 0.05, sig, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            start_local, end_local = dt_run_utc.astimezone(rome_tz), (dt_run_utc + timedelta(hours=max_h)).astimezone(rome_tz)
            plt.title(f"ICON-D2 EPS - Precipitazione Cumulata Totale (mm)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {start_local.strftime('%H:%M del %d/%m/%Y')} - {end_local.strftime('%H:%M del %d/%m/%Y')} (+{max_h}h)", fontweight='bold')
            filename = f"accum_cum_{idx}.png"
            plt.savefig(filename, dpi=200, bbox_inches='tight'); plt.close(fig); percorsi_foto.append(filename)
        except Exception as e: print(f"  ❌ Errore data {date_str}: {e}")

    if percorsi_foto:
        invia_album_telegram(percorsi_foto, f"ICON-D2 EPS: Precipitazione Cumulata Totale (mm)\nRun {nome_run} ({dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')})")
        for f in percorsi_foto:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    data = fetch_dati_con_retry()
    if data:
        is_new, nome_run, dt_run_utc = estrai_limiti_run(data.get("hourly", {}), "temperature_2m", data.get("utc_offset_seconds", 0))
        if is_new: genera_album_cumulativo(dt_run_utc, nome_run)
