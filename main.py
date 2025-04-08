import xarray as xr
import numpy as np
import pandas as pd
import  geopandas as gpd
from datetime import datetime, timedelta
import aiohttp
import asyncio
from aiohttp import ClientTimeout
import os
from rasterstats import zonal_stats
from scipy.spatial import cKDTree
from tqdm import tqdm
import tempfile
import rasterio
from rasterio.transform import from_origin

from dotenv import load_dotenv
import os

import json


import psycopg2

load_dotenv()

conn = psycopg2.connect(
    dbname=os.environ.get('DATABASE_NAME'), 
    user=os.environ.get('DATABASE_USER'), 
    password=os.environ.get('DATABASE_PASSWORD'), 
    host=os.environ.get('DATABASE_HOST'), 
    port=os.environ.get('DATABASE_PORT')
)

cursor = conn.cursor()

DF_CITIES = {}
PARAMETERS = {}

async def download_file(url, filename, semaphore):
    try:
        async with semaphore:
            timeout = ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        with open(filename, "wb") as f:
                            while chunk := await response.content.read(1024 * 1024):
                                f.write(chunk)
                        print(f"✅ Download concluído: {filename}")
                    else:
                        print(f"⚠️ Erro no download ({response.status}): {url}")
    except Exception as e:
        print(f"❌ Erro inesperado em {url}: {e}")

async def baixar_grib_hoje():
    hoje = datetime.now() - timedelta(days=1)
    ano, mes, dia = hoje.year, hoje.month, hoje.day
    url = f"https://ftp.cptec.inpe.br/modelos/tempo/MERGE/GPM/DAILY/{ano}/{mes:02}/MERGE_CPTEC_{ano}{mes:02}{dia:02}.grib2"
    filename = 'grib_dia.grib2'


    # Baixar somente se não existir
    if not os.path.exists(filename):
        semaphore = asyncio.Semaphore(1)
        await download_file(url, filename, semaphore)
    else:
        print("📂 Arquivo já existe localmente.")

    return filename, hoje

def getCities():
    cursor.execute(f"SELECT * FROM cities where id != 2")

    results = cursor.fetchall()

    return pd.DataFrame.from_records(results, columns=[col[0] for col in cursor.description])

def getParameters():
    cursor.execute(f"SELECT * FROM parameters where parameter_type_id = 5")

    results = cursor.fetchall()

    return pd.DataFrame.from_records(results, columns=[col[0] for col in cursor.description])

def saveParameter(city_id, dsc):
    
    sql = f"""
        INSERT INTO parameters (name, parameterizable_type, parameterizable_id, values, created_at, updated_at, parameter_type_id) VALUES ('NewCityParameter', 'City', {city_id}, '{{"climate": {{"dsc": {dsc}}}}}', now(), now(), 5) ON CONFLICT (parameterizable_type, parameterizable_id, parameter_type_id) DO UPDATE SET 
        values = (parameters.values::jsonb || excluded.values::jsonb)::json,
        updated_at = now()
    """
    
    cursor.execute(sql)

    conn.commit()


DF_CITIES = getCities()
PARAMETERS = getParameters()

# print(PARAMETERS['values'].apply(pd.read_json))




filename_hoje, data_hoje = asyncio.run(baixar_grib_hoje())
print(filename_hoje)
ds =  xr.open_dataset(filename_hoje)
municipios_sp = gpd.read_file('zonal/municipios_sp.shp', encoding='utf-8')
municipios_sp = municipios_sp.to_crs("EPSG:4326")
# Define resolução (assume 0.1° que é típico no MERGE, ajuste se necessário)
res = 0.1

# Define o transform com base no canto superior esquerdo
transform = from_origin(
    west=ds.longitude.min().item() - res / 2,
    north=ds.latitude.max().item() + res / 2,
    xsize=res,
    ysize=res
)

# Extrair grade do raster
latitudes = ds.latitude.values
longitudes = ds.longitude.values
lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
coords = np.column_stack((lon_grid.ravel(), lat_grid.ravel()))

# Criar árvore de busca para coordenadas
tree = cKDTree(coords)

# Inicializar lista de resultados
resultados = []

data = pd.to_datetime(str(ds.time.values)).date() if "time" in ds.coords else None

array = ds.prec.values.astype(np.float32)

import os
import tempfile

# Defina o caminho para salvar o arquivo temporário manualmente
tmp_path = os.path.join(os.getcwd(), "temp_raster.tif")

with rasterio.open(
    tmp_path,
    "w",
    driver="GTiff",
    height=array.shape[0],
    width=array.shape[1],
    count=1,
    dtype="float32",
    crs="EPSG:4326",
    transform=transform,
    nodata=np.nan
) as dst:
    dst.write(array, 1)

# Depois você pode excluir esse arquivo manualmente se quiser

    for idx, row in municipios_sp.iterrows():
        # Estatística zonal
        stat = zonal_stats(
            [row['geometry']],
            "temp_raster.tif",
            stats=["max"],
            nodata=np.nan
        )[0]["max"]

        if stat is None or pd.isna(stat):
            centroide = row.geometry.centroid
            dist, idx_nearest = tree.query([centroide.x, centroide.y])
            lat_idx, lon_idx = np.unravel_index(idx_nearest, lon_grid.shape)
            stat = array[lat_idx, lon_idx]

        resultados.append({
            "cd_mun": row["cd_mun"],
            "data": data,
            "prec_max": stat
        })


# Converter para DataFrame
df_prec_max = pd.DataFrame(resultados)

hoje = datetime.now() - timedelta(days=1)
ano, mes, dia = hoje.year, hoje.month, hoje.day

ontem = datetime.now() - timedelta(days=2)
ano_o, mes_o, dia_o = ontem.year, ontem.month, ontem.day

city_ids = DF_CITIES['id'].unique()
df_list = []

for id in city_ids:
    city = DF_CITIES[DF_CITIES['id'] == id].iloc[0]

    parameter = PARAMETERS[PARAMETERS['parameterizable_id'] == id]

    if(len(parameter) > 0):
        parameter = parameter.iloc[0]
    
        dsc = parameter['values']['climate']['dsc']
        df_list.append({'cd_mun': city['cod_ibge'], 'DSC': dsc })
    
df_dias_secos = pd.DataFrame(df_list)


# df_dias_secos = pd.read_csv(f'ds_dsc{ano_o}{mes_o:02}{dia_o:02}.csv')




# df_dias_secos["cd_mun"] = df_dias_secos["cd_mun"].astype(str)
df_prec_max["cd_mun"] = df_prec_max["cd_mun"].astype(str)
df_atualizado = pd.merge(df_dias_secos, df_prec_max[['cd_mun', 'prec_max']], on='cd_mun', how='left')

# Aplicar as regras para atualização
def atualizar_dias_secos(row):
    if row['prec_max'] < 1:
        # row['DS'] += 1
        row['DSC'] += 1
    else:
        # DS mantém, DSC zera
        row['DSC'] = 0
    return row

# Aplicar função linha a linha
df_atualizado = df_atualizado.apply(atualizar_dias_secos, axis=1)

# Atualizar o DataFrame original
df_dias_secos_new = df_atualizado[['cd_mun',  'DSC']].copy()

cds = df_dias_secos_new['cd_mun'].unique()

# ibge = cds[0]
# print(first)
# for ibge in cds:
#     dsc = df_dias_secos_new[df_dias_secos_new['cd_mun'] == ibge].iloc[0]['DSC']
#     id = DF_CITIES[DF_CITIES['cod_ibge'] == ibge].iloc[0]['id']
#     # print(id, ibge)
#     print(f'salvando {ibge} {id} {dsc}')
#     saveParameter(id, dsc)

# df_dias_secos_new.to_csv(f'ds_dsc{ano}{mes:02}{dia:02}.csv', index=False)

