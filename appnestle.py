### Librerías

from PIL import Image
import streamlit as st
import plotly.express as px
import pandas as pd
from dateutil.relativedelta import *
import seaborn as sns; sns.set_theme()
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

path_favicon = r'./img/favicon1.ico'
im = Image.open(path_favicon)
st.set_page_config(page_title='CM AI27', page_icon=im, layout="wide")
path = r'./img/AI27 Logo1.png'
image = Image.open(path)
col1, col2, col3 = st.columns([1,2,1])
col2.image(image, use_column_width=True)

@st.cache_data(show_spinner='Cargando Datos... Espere...', persist=True)
def load_df():
    
    rEmbarques = r'./data/Salidas Nestle 2023.xlsx'
    
    Embarques = pd.read_excel(rEmbarques, sheet_name = "Data")

    Embarques['Inicio'] = pd.to_datetime(Embarques['Inicio'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    Embarques['Arribo'] = pd.to_datetime(Embarques['Arribo'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    Embarques['Finalización'] = pd.to_datetime(Embarques['Finalización'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    Embarques.Arribo.fillna(Embarques.Finalización, inplace=True)
    Embarques['TiempoCierreServicio'] = (Embarques['Finalización'] - Embarques['Arribo'])
    Embarques['TiempoCierreServicio'] = Embarques['TiempoCierreServicio']/np.timedelta64(1,'h')
    Embarques['TiempoCierreServicio'].fillna(Embarques['TiempoCierreServicio'].mean(), inplace=True)
    Embarques['TiempoCierreServicio'] = Embarques['TiempoCierreServicio'].astype(int)
    Embarques['Destinos'].fillna('OTRO', inplace=True)
    Embarques['Línea Transportista'].fillna('OTRO', inplace=True)
    Embarques['Duración'].fillna(Embarques['Duración'].mean(), inplace=True)
    Embarques['Duración'] = Embarques['Duración'].astype(int)
    Embarques['Año'] = Embarques['Inicio'].apply(lambda x: x.year)
    Embarques['Mes'] = Embarques['Inicio'].apply(lambda x: x.month)
    Embarques['DiadelAño'] = Embarques['Inicio'].apply(lambda x: x.dayofyear)
    Embarques['SemanadelAño'] = Embarques['Inicio'].apply(lambda x: x.weekofyear)
    Embarques['DiadeSemana'] = Embarques['Inicio'].apply(lambda x: x.dayofweek)
    Embarques['Quincena'] = Embarques['Inicio'].apply(lambda x: x.quarter)
    Embarques['MesN'] = Embarques['Mes'].map({1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril", 5:"Mayo", 6:"Junio", 7:"Julio", 8:"Agosto", 9:"Septiembre", 10:"Octubre", 11:"Noviembre", 12:"Diciembre"})
    Embarques['Origen Destino'] = Embarques['Estado Origen'] + '-' + Embarques['Estado Destino']
    Embarques = Embarques.dropna() 
    return Embarques

st.cache_data(ttl=3600)
def crear_df(df):

    df['Duración'] = df['Duración'].astype(int)
    df1 = pd.DataFrame(data = df, index = np.repeat(df.index, abs(df['Duración'])))
    df2 = pd.DataFrame(pd.concat(
                [
                    df.iloc[:,0].reindex(
                    pd.date_range(idx, periods=abs(row["Duración"]), freq="H"),
                    )
                    for idx, row in df.iterrows()
                ]).ffill())
    
    df2["HorasActiva"] = df2.index
    df2.drop(['Bitácora'], axis = 'columns', inplace=True)        
    df2.reset_index(drop=True, inplace=True)
    df2['HorasActiva'] = [d.time() for d in df2['HorasActiva']]
    df2['HorasActiva'] = pd.to_timedelta(df2['HorasActiva'].astype(str))
    df1.reset_index(drop=True, inplace=True)
    df1['Inicio'] = pd.to_datetime(df1['Inicio'].astype(str), format='%Y-%m-%d %H:%M:%S', errors='coerce')
    dfs = pd.concat([df1, df2], axis=1)
    dfs["FHBitacora"] = dfs["Inicio"] + dfs["HorasActiva"]
    #dfs["FHBitacora"] = dfs["Inicio"].astype(str) + dfs["HorasActiva"].astype(str)
    #dfs["FHBitacora"] = dt.datetime.strptime(dfs["Inicio"],'%Y-%m-%d %H:%M:%S') + dt.datetime.strptime(dfs["HorasActiva"],'%Y-%m-%d %H:%M:%S')
    # Reindexar el dfframe
    #dfs["FHBitacora"] =  pd.to_datetime(dfs["FHBitacora"], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    dfs.index = dfs["FHBitacora"] 
    #Eliminar Columnas que no son parte del Análisis
    dfs.drop(['Cliente', 'Tipo Monitoreo','Total Anomalías','Calificación','Origen','Destinos','Línea Transportista','Tipo Unidad','Inicio','Arribo','Finalización','Tiempo Recorrido','Duración', 'Robo', 'Mes'], axis = 'columns', inplace=True)
    #Remuestreo
    dfs = dfs.Bitácora.resample('H').count()
    #Generar dfFrame
    dfs = pd.DataFrame(dfs)
    return dfs

st.cache_data(ttl=3600)
def servicios_personal(df):

    sr_data1 = go.Scatter(x = df.index,
                        y=df['Bitácora'],
                        line=go.scatter.Line(color='red', width = 0.6),
                        opacity=0.8,
                        yaxis = 'y1',
                        hoverinfo = 'text',
                        name='Servicios/Hora',
                        text= [f'Servicios: {x:.0f} por Hora' for x in df['Bitácora']])
    
    sr_data2 = go.Scatter(x = df.index,
                        y=df['Personal Requerido'],
                        opacity=0.8,
                        yaxis = 'y2',
                        name='Personal Requerido/Hora',
                        text= [f'Personal Requerido: {x:.0f} por Hora' for x in df['Personal Requerido']]
                        )

    # Create a layout with interactive elements and two yaxes
    layout = go.Layout(height=700, width=1400, font=dict(size=10),
                   plot_bgcolor="#FFF",
                   xaxis=dict(showgrid=False, title='Fecha', hovermode="x unified",
                                        # Range selector with buttons
                                         rangeselector=dict(
                                             # Buttons for selecting time scale
                                             buttons=list([
                                                 # 1 month
                                                 dict(count=1,
                                                      label='1m',
                                                      step='month',
                                                      stepmode='backward'),
                                                 # 1 week
                                                 dict(count=7,
                                                      label='1w',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 1 day
                                                 dict(count=1,
                                                      label='1d',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 12 hours
                                                 dict(count=12,
                                                      label='12h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # 4 hours
                                                 dict(count=4,
                                                      label='4h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # 1 hour
                                                 dict(count=1,
                                                      label='1h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # Entire scale
                                                 dict(step='all')
                                             ])
                                         ),
                                         # Sliding for selecting time window
                                         rangeslider=dict(visible=True),
                                         # Type of xaxis
                                         type='date'),
                   yaxis=dict(showgrid=False, title='Personal Requerido/Hora', color='red', side = 'left'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(showgrid=False, title='Servicios/Hora', color='blue',
                                          overlaying='y1',
                                          side='right')
                   )
    fig = go.Figure(data=[sr_data1, sr_data2], layout=layout)
    st.plotly_chart(fig)

st.cache_data(ttl=3600)
def servicios_desfase(df):

        sr_data3 = go.Scatter(x = df.index,
                        y=df['Personal Requerido'],
                        line=go.scatter.Line(color='red', width = 0.6),
                        opacity=0.8,
                        yaxis = 'y1',
                        hoverinfo = 'text',
                        name='Personal Requerido/Hora',
                        text= [f'Personal Requerido: {x:.0f} por Hora' for x in df['Personal Requerido']])
    
        sr_data4 = go.Scatter(x = df.index,
                        y=df['Desfase'],
                        opacity=0.8,
                        yaxis = 'y2',
                        name='Tiempo de Desfase',
                        text= [f'Desfase: {x:.0f} minutos' for x in df['Desfase']]
                        )

        # Create a layout with interactive elements and two yaxes
        layout = go.Layout(height=700, width=1400, font=dict(size=10), hovermode="x unified",
                plot_bgcolor="#FFF",
                xaxis=dict(showgrid=False, title='Fecha',
                                        # Range selector with buttons
                                         rangeselector=dict(
                                             # Buttons for selecting time scale
                                             buttons=list([
                                                 # 1 month
                                                 dict(count=1,
                                                      label='1m',
                                                      step='month',
                                                      stepmode='backward'),
                                                 # 1 week
                                                 dict(count=7,
                                                      label='1w',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 1 day
                                                 dict(count=1,
                                                      label='1d',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 12 hours
                                                 dict(count=12,
                                                      label='12h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # 4 hours
                                                 dict(count=4,
                                                      label='4h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # 1 hour
                                                 dict(count=1,
                                                      label='1h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # Entire scale
                                                 dict(step='all')
                                             ])
                                         ),
                                         # Sliding for selecting time window
                                         rangeslider=dict(visible=True),
                                         # Type of xaxis
                                         type='date'),
                   yaxis=dict(showgrid=False, title='Personal Requerido/Hora', color='red', side = 'left'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(showgrid=False, title='Desfase/Hora', color='blue',
                                          overlaying='y1',
                                          side='right')
                   )
        fig = go.Figure(data=[sr_data3, sr_data4], layout=layout)
        st.plotly_chart(fig)

def carga_trabajo(df):

            sr_data3 = go.Bar(x = df.index,
                        y=df['Bitácora'],
                        opacity=0.8,
                        yaxis = 'y1',
                        marker= dict(color = "gray"),
                        hoverinfo = 'text',
                        name='Servicios/Hora',
                        )
    
            sr_data4 = go.Scatter(x = df.index,
                        y=df['Desfase'],
                        line=go.scatter.Line(color='orange', width = 3),
                        opacity=0.8,
                        yaxis = 'y2',
                        name='Desfase/Hora',
                        text= [f'Desfase: {x:.0f} Horas' for x in df['Desfase']]
                        )
            
            sr_data5 = go.Scatter(x = df.index,
                        y=df['Personal Requerido'],
                        line=go.scatter.Line(color='green', width = 2),
                        opacity=0.8,
                        yaxis = 'y2',
                        name='Personal Requerido/Hora',
                        text= [f'Personal Requerido: {x:.0f} por Hora' for x in df['Personal Requerido']]
                        )

            # Create a layout with interactive elements and two yaxes
            layout = go.Layout(height=700, width=1400, font=dict(size=10), hovermode="x unified",
                   plot_bgcolor="#FFF",
                   xaxis=dict(showgrid=False, title='Fecha',
                                        # Range selector with buttons
                                         rangeselector=dict(
                                             # Buttons for selecting time scale
                                             buttons=list([
                                                 # 1 month
                                                 dict(count=1,
                                                      label='1m',
                                                      step='month',
                                                      stepmode='backward'),
                                                 # 1 week
                                                 dict(count=7,
                                                      label='1w',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 1 day
                                                 dict(count=1,
                                                      label='1d',
                                                      step='day',
                                                      stepmode='todate'),
                                                 # 12 hours
                                                 dict(count=12,
                                                      label='12h',
                                                      step='hour',
                                                      stepmode='backward'),
                                                 # Entire scale
                                                 dict(step='all')
                                             ])
                                         ),
                                         # Sliding for selecting time window
                                         rangeslider=dict(visible=True),
                                         # Type of xaxis
                                         type='date'),
                   yaxis=dict(showgrid=False, title='Servicios Activos/Hora', color='red', side = 'left'),
                   # Add a second yaxis to the right of the plot
                   yaxis2=dict(showgrid=False, title='Desfase/Hora', color='blue',
                                          overlaying='y1',
                                          side='right')
                   )
            fig = go.Figure(data=[sr_data3, sr_data4, sr_data5], layout=layout)
            st.plotly_chart(fig)

try:
    
    df = load_df()

    st.markdown("<h2 style='text-align: left;'>Carga de Trabajo Operativa del Centro de Monitoreo para Nestlé</h2>", unsafe_allow_html=True)

    st.write(f"Este módulo contiene información histórica de salidas desde **{df.MesN.values[0]} {df.Año.values[0].astype(int)}** a **{df.MesN.values[-1]} {df.Año.values[-1].astype(int)}** :")

    st.write(""" 
    La finalidad de este módulo es consultar el histórico de salidas de Nestlé y realizar seguimiento a la carga de trabajo. Pasos a seguir para este módulo:
    1. Seleccionar el **División** del cuál desea obtener información de salidas y carga de trabajo. Seleccionando el checkbox, puede seleccionar todas las divisiones de Nestlé.
    2. Seleccionar el **Mes** del cuál desea obtener información de salidas y carga de trabajo. Seleccionando el checkbox, puede seleccionar ttodas las divisiones de Nestlé.
       """)

    x1, x2 = st.columns(2)

    with x1:
        containerC1 = st.container()
        allC1 = st.checkbox("Seleccionar Todos", key="FF")
        if allC1: 
             sorted_unique_cliente = sorted(df['Cliente'].unique())
             selected_cliente = containerC1.multiselect('División(es):', sorted_unique_cliente, sorted_unique_cliente, key="FF1")
             df_selected_cliente = df[df['Cliente'].isin(selected_cliente)].astype(str)
        else:
            sorted_unique_cliente = sorted(df['Cliente'].unique())
            selected_cliente = containerC1.multiselect('División(es)', sorted_unique_cliente, key="FF1")
            df_selected_cliente = df[df['Cliente'].isin(selected_cliente)].astype(str)

    with x2:
        containerTS1 = st.container()
        allTS1 = st.checkbox("Seleccionar Todos", key="GG")
        if allTS1:
            sorted_unique_mes = sorted(df_selected_cliente['MesN'].unique())
            selected_mes = containerTS1.multiselect('Mes(es):', sorted_unique_mes, sorted_unique_mes, key="GG1") 
            df_selected_dia = df_selected_cliente[df_selected_cliente['MesN'].isin(selected_mes)].astype(str)
        else:
            sorted_unique_mes = sorted(df_selected_cliente['MesN'].unique())
            selected_mes = containerTS1.multiselect('Mes(es):', sorted_unique_mes, key="GG1") 
            df_selected_dia = df_selected_cliente[df_selected_cliente['MesN'].isin(sorted_unique_mes)].astype(str)
    
    #### Módulo Gráfico Histórico

    # Grafico de Servicios vs Personal Requerido
    df1 = crear_df(df_selected_dia)    
    df2 = df1.copy()
    df2['Personal Requerido'] = np.ceil(df2['Bitácora'].astype(int)/25) # El 25 es la cantidad de bitacoras por hora con atención de 2.5 minutos
    st.markdown("<h3 style='text-align: left;'>Servicios Activos por Hora - Personal Requerido por hora</h3>", unsafe_allow_html=True)

    grafica_servicios_personal = servicios_personal(df2)

    # Grafico de Servicios vs Desfases de Atención

    st.markdown("<h3 style='text-align: left;'>Servicios Activos por Hora - Tiempo del Desfase de Atención</h3>", unsafe_allow_html=True)

    df3 = crear_df(df_selected_dia)
    df4 = df3.copy()
    n_estaciones = st.slider('Nro de Estaciones:', 1, 20)
    recursos = (n_estaciones * 25) * 2.5

    df4['Personal Requerido'] = np.ceil(df4['Bitácora'].astype(int)/25) # El 25 es la cantidad de bitacoras por hora con atención de 2.5 minutos
    df4['Calculo de Desfase'] = (np.ceil(df4['Bitácora'].astype(int) * 2.5) - recursos)
    df4['Desfase'] = df4['Calculo de Desfase'].map(lambda x: x-60 if x > 60 else 0)

    grafica_servicios_desfase = servicios_desfase(df4)


    st.markdown("<h3 style='text-align: left;'>Servicios Activos por Hora - Tiempo del Desfase de Atención - Personal requerido</h3>", unsafe_allow_html=True)
    grafica_servicios_desfase_personal_requerido = carga_trabajo(df4)

except NameError as e:
    print("Seleccionar: ", e)

except ZeroDivisionError as e:
    print("Seleccionar: ", e)
    
except KeyError as e:
    print("Seleccionar: ", e)

except ValueError as e:
    print("Seleccionar: ", e)
    
except IndexError as e:
    print("Seleccionar: ", e)

    # ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

