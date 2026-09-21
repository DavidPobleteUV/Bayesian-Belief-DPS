# Robust DPS con ε-NSGA-II — guía para correr en el servidor

Este documento se lee **en el servidor**, después de clonar o actualizar el
repo. Contiene por qué existe esta variante, cómo lanzarla, qué medir antes de
comprometer días de máquina, y qué limitaciones hay que declarar al reportar
los resultados.

La versión NSGA-II (`weap_dps/main_robust_weap.py`, `run_robust_server.ps1`)
**queda intacta**. Las dos conviven a propósito: la comparación entre ambas es
el objeto del experimento.

---

## 1. Por qué esta variante

En la corrida iter02 (5 semillas × 4.000 evaluaciones × 27 estados del mundo,
65 h de reloj) las cinco semillas devolvieron un **frente de 100 sobre una
población de 100**. Es decir: la población entera mutuamente no dominada.

Con 5 objetivos el rango de Pareto pierde poder discriminante. Si ninguna
solución domina a ninguna otra, todas quedan en rango 1 y **la presión de
selección por dominancia es exactamente nula**: lo único que decide quién
sobrevive es la distancia de apiñamiento, que es una medida de diversidad y no
de convergencia. El algoritmo deja de empujar hacia el frente verdadero y
empieza a repartir puntos.

El archivo de **ε-dominancia** **no actúa sobre esa presión directamente**, y
conviene decirlo con precisión porque una versión anterior de esta guía afirmaba
lo contrario. En la implementación de Platypus, la selección de padres y la
truncación de la población siguen exactamente las reglas de NSGA-II: los padres
salen de la población, y el archivo solo **recibe** soluciones —nunca las entrega
a la selección—. Influye en la búsqueda por una única vía, **los reinicios**:
cuando se disparan, la población se reconstruye a partir del archivo más
mutantes y se redimensiona a unas cuatro veces su tamaño.

Lo que el archivo sí aporta, y es por lo que vale la pena:

- **un frente con resolución declarada.** Cada ε dice cuánta diferencia es
  significativa en cada objetivo, y dos políticas que difieren en menos quedan
  en la misma casilla. El frente deja de ser «las 100 que sobrevivieron» para
  ser «lo mejor encontrado, a la resolución que importa para decidir»;
- **un tamaño acotado** sin recurrir al apiñamiento;
- **memoria de lo mejor encontrado**, que los reinicios usan para reinyectar
  diversidad cuando la búsqueda se estanca;
- si J1 y J6 se reincorporaran como objetivos, su variación de 1,8 % y 0,2 %
  colapsaría a una o dos casillas del archivo, sin inflarlo por accidente
  dimensional. Hoy siguen fuera del conjunto optimizado (§8).

**Si ε-NSGA-II mejora o no la presión de selección en la población es una
pregunta empírica**, y se responde midiendo la fracción no dominada de la
población a igual número de evaluaciones (§7), no por construcción.

Además hay una segunda pregunta que la corrida anterior **no puede responder**:
si 4.000 evaluaciones alcanzaron. El CV del hipervolumen entre semillas era
1,3 %, pero eso acota la varianza entre semillas, no el sesgo común a todas —
cinco semillas pueden converger consistentemente a la misma región subóptima.
Lo que falta es la **trayectoria** del hipervolumen contra el número de
evaluaciones, y por eso esta versión la registra.

---

## 2. Qué se agregó

| archivo | qué hace |
|---|---|
| `weap_dps/main_eps_robust_weap.py` | ε-NSGA-II: archivo ε, reinicios adaptativos, registro de HV(nfe) |
| `weap_dps/hv_utils.py` | hipervolumen estimado por Monte Carlo sobre caja fija |
| `weap_dps/comparar_algoritmos.py` | NSGA-II contra ε-NSGA-II y curva de convergencia |
| `weap_dps/benchmark_eval.py` | costo real por evaluación en la máquina donde se corra |
| `run_eps_server.ps1` | lanzador, hermano de `run_robust_server.ps1` |
| `weap_dps/config_weap.py` | `EPSILONS_BY_OBJECTIVE`, `HV_MINIMUM`, `HV_MAXIMUM` |

`main_eps_robust_weap.py` **importa** `RobustPipeWEAP` de la versión NSGA-II en
lugar de copiarlo. Si se copiara, cualquier divergencia futura —una
calibración, un umbral— haría que la comparación midiera dos cosas distintas
creyendo medir el algoritmo.

### Los ε

Calibrados sobre el rango real de la unión de los cinco frentes de iter02, de
modo que cada eje quede con resolución comparable. Si un ε fuera mucho más fino
que los otros, ese objetivo dominaría el tamaño del archivo.

| objetivo | rango del frente iter02 | ε | casillas |
|---|---|---|---|
| J2 déficit AP | 3,80e6 – 2,08e7 m³ | 2,5e5 m³ | ~68 |
| J3 valor agrícola | 9,83e9 CLP (~10,0 MUSD) | 2,0e8 CLP | ~49 |
| J4 costo suministro | 7,06e10 CLP (~72 MUSD) | 1,0e9 CLP | ~71 |
| J51 semanas en falla | 64 – 506 | 10 semanas | ~44 |
| J52 déficit peor año | 0,116 – 0,683 | 0,01 | ~57 |

Cada ε es **la menor diferencia que cambiaría una decisión** en ese objetivo.
Es un juicio del analista, no un parámetro técnico: si alguno no te parece,
edítalo en `config_weap.py`. Para barrer la resolución sin editar el
diccionario está `DPS_EPS_SCALE` (`2` = archivo la mitad de fino, converge
antes) o el parámetro `-EpsScale` del lanzador.

---

## 3. Referencia contra la que se compara

Hipervolumen de la corrida NSGA-II de iter02, calculado con la caja fija de
`config_weap.py`:

| | |
|---|---|
| HV medio | **0,72720** (CV 0,7 % entre semillas) |
| HV de la unión de las 5 semillas | **0,75637** |
| fracción no dominada de la población, a 4.000 evaluaciones | **1,00 en las cinco** (en NSGA-II el frente reportado es la población final) |
| soluciones fuera de la caja de HV | 0 |

Se reproduce en cualquier máquina con:

```powershell
.\venv_DPS\Scripts\python.exe weap_dps\comparar_algoritmos.py
```

---

## 4. Antes de lanzar: medir el costo real

El ETA que traen los lanzadores (2,17 s por escenario, 58,5 s por evaluación de
27 escenarios) está **medido en la PC de trabajo**: 65,0 h / 4.000 evaluaciones
/ 27 escenarios. Extrapolarlo a otra máquina es adivinar.

Lo que manda es **la velocidad de un hilo**, no el número de cores, porque cada
semilla corre con `OMP_NUM_THREADS=1`. Un servidor con 64 cores pero hilos
lentos puede ser *más* lento por semilla.

```powershell
.\venv_DPS\Scripts\python.exe weap_dps\benchmark_eval.py --n 3 --par 5
```

Tarda unos minutos e imprime el reloj proyectado para 4.000, 10.000 y 20.000
evaluaciones, medido con un proceso y con N en paralelo. Lo segundo importa y no
es un detalle: en la PC de trabajo, una semilla sola cuesta 41,9 s por
evaluación y cinco en paralelo cuestan 58,5 s cada una — **+40 %**. Un ETA
medido con un solo proceso subestima el reloj en esa proporción. Usa como `--par`
el número de semillas que piensas lanzar.

Datos de la máquina, para dimensionar cuántas semillas caben:

```powershell
$cs = Get-CimInstance Win32_ComputerSystem; $cpu = Get-CimInstance Win32_Processor; [PSCustomObject]@{ CPU=$cpu.Name; Fisicos=$cpu.NumberOfCores; Logicos=$cs.NumberOfLogicalProcessors; RAM_GB=[math]::Round($cs.TotalPhysicalMemory/1GB,1); DiscoLibre_GB=[math]::Round((Get-PSDrive C).Free/1GB,1) } | Format-List
```

### Costo medido en la PC de trabajo (12 cores lógicos)

| condición | s/escenario | s/evaluación (27 esc.) |
|---|---|---|
| **1 proceso solo** | 1,55 | **41,9** |
| **5 semillas en paralelo** | 2,17 | **58,5** |

La competencia entre semillas cuesta **+40 % por semilla**, y no es despreciable:
un ETA medido con un solo proceso subestima el reloj en ese 40 %. Por eso el
benchmark se corre con `--par N`, no solo.

**RAM medida: 0,55 GB por proceso** (working set en régimen, 27 escenarios,
modelo iter02 cargado). El `0,9 + 0,005 × n_escenarios` que usaba
`run_robust_server.ps1` era una estimación declarada como no medida y
sobrestimaba ~2×, de modo que en una máquina chica desaconsejaba semillas que sí
caben. `run_eps_server.ps1` usa 0,66 GB (lo medido más 20 % de margen) y reserva
2,5 GB para el sistema operativo.

**Conclusión de dimensionamiento: el cuello de botella es CPU, no RAM.** Cada
semilla ocupa un core entero (`OMP_NUM_THREADS=1`) y medio giga.

### Cuánto demora

**Las semillas son gratis y las evaluaciones no.** El reloj lo fija el
presupuesto *por semilla*, no cuántas semillas corras — mientras quepan en los
cores disponibles.

| presupuesto por semilla | reloj en la PC de trabajo |
|---|---|
| 4.000 evaluaciones (comparación directa con lo existente) | ~65 h ≈ 2,7 días |
| 10.000 evaluaciones | ~163 h ≈ 6,8 días |

Con 10.000 y checkpoints, el corte de 4.000 queda registrado en la curva HV(nfe)
sin costo adicional, así que se puede abortar si ya se aplanó.

### El servidor es más chico que la PC de trabajo

Medido el 20-09-2026:

| | PC de trabajo | servidor |
|---|---|---|
| cores lógicos | 12 | **8** |
| RAM | — | **7,9 GB** |
| CPU | — | `Common KVM processor` (virtualizado, modelo enmascarado) |
| disco libre | — | 298 GB |

Consecuencias, y conviene tenerlas claras antes de comprometer la máquina:

1. **Caben hasta 7 semillas** (8 cores menos uno para el sistema). La RAM
   alcanza para ~8, así que no es ella la que limita.
2. **No se pueden correr los dos experimentos a la vez.** Con 12 cores la idea
   era lanzar ε-NSGA-II y NSGA-II con presupuesto extendido en paralelo, para
   separar el efecto del algoritmo del de gastar más evaluaciones. Con 8 cores
   hay que elegir, o correrlos en serie.
3. **No entrenar `iter02_base` ni `iter02_wide` en la misma ventana.** El
   entrenamiento del MLP usa varios hilos y le quitaría cores a las semillas,
   alargando el reloj de ambas cosas.
4. **`Common KVM processor` significa que la velocidad de un hilo es
   desconocida**, y es justamente lo que fija el reloj. Puede ser más lento que
   la PC de trabajo, en cuyo caso 10.000 evaluaciones pasan de 6,8 días a algo
   bastante peor. **Correr el benchmark antes de decidir el presupuesto no es
   opcional en esta máquina.**

### Resultado del benchmark en el servidor: descartado para esta corrida

Medido el 20-09-2026 con `benchmark_eval.py --n 3 --par 5`:

| | PC de trabajo | servidor | relación |
|---|---|---|---|
| s/evaluación, 1 proceso | 41,9 | **73,0** | 1,74× más lento |
| s/evaluación, 5 en paralelo | 58,5 | **182,5** | **3,12× más lento** |
| penalización por competencia | +40 % | **+150 %** | — |

| presupuesto | reloj en el servidor |
|---|---|
| 4.000 evaluaciones | **8,4 días** |
| 10.000 evaluaciones | **21,1 días** |

La penalización de +150 % es el dato revelador: cinco procesos de **un solo
hilo** sobre 8 cores lógicos no deberían competir así. Indica que los vCPU del
KVM están **sobresuscritos a nivel del hipervisor** — la máquina comparte CPU
física con otros huéspedes, de modo que los "8 cores" no son 8 cores
dedicados. Ningún ajuste del lado del DPS corrige eso.

**Conclusión: el Robust DPS se corre en la PC de trabajo.** El servidor queda
para el entrenamiento del MLP (`iter02_base`, `iter02_wide`), que es carga de
otro tipo y donde perder tiempo de reloj cuesta menos.

Antes de comprometer el servidor a entrenar, conviene verificar si tiene GPU:
la misma sobresuscripción de vCPU afectaría a un entrenamiento por CPU.

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

---

## 5. Requisito que el `git pull` NO trae

**`data_weap_iter02/` está en `.gitignore`** (24 MB: checkpoint, scalers,
template, manifiesto). Hay que copiarlos a mano al servidor:

```
data_weap_iter02\best_model.ckpt
data_weap_iter02\X_template.npz
data_weap_iter02\scalers_weap.npz
data_weap_iter02\transform_params_weap.npz
data_weap_iter02\manifest_inputs.csv
```

`run_eps_server.ps1` los verifica al arrancar y falla en el primer segundo si
faltan. Eso es deliberado: sin la verificación, `config_weap` caería a
`data_weap/` —el emulador de iter01— y la corrida saldría con el modelo
equivocado **sin dar ningún error**.

El lanzador fija `DPS_DATA_DIR` apuntando a `data_weap_iter02`.

---

## 6. Cómo lanzar

```powershell
git pull
.\run_eps_server.ps1
```

Por defecto: 10.000 evaluaciones, 5 semillas (42, 123, 456, 789, 1010),
población inicial 100 con tope 300, ventana de reinicio 10 generaciones, salida
en `runs_weap\eps_iter02`.

Variantes:

```powershell
.\run_eps_server.ps1 -Evaluations 4000
```

```powershell
.\run_eps_server.ps1 -Seeds 42,123,456 -EpsProgress
```

`-EpsProgress` cambia la continuación temporal por **continuación por
ε-progreso**: reinicia cuando el archivo deja de mejorar, que es el mecanismo
de Borg. Está disponible pero **no es el default**, para que la comparación se
haga contra un ε-NSGA-II estándar y no contra un híbrido.

### Parámetros del lanzador

| parámetro | default | qué hace |
|---|---|---|
| `-Evaluations` | 10000 | presupuesto por semilla |
| `-Population` | 100 | población **inicial**; el algoritmo la reescala |
| `-Seeds` | 42,123,456,789,1010 | semillas, en paralelo |
| `-RestartWindow` | 10 | generaciones entre chequeos de reinicio |
| `-MaxPopulation` | 300 | tope de población tras un reinicio |
| `-EpsScale` | 1.0 | multiplica todos los ε |
| `-EpsProgress` | (apagado) | continuación por ε-progreso |
| `-OutDir` | `runs_weap\eps_iter02` | salida |

Por qué `-RestartWindow 10` y no el default de Platypus: Platypus chequea cada
**100 generaciones**. Con población 100 y 10.000 evaluaciones hay 100
generaciones en total, así que el chequeo ocurriría una sola vez, al final: el
mecanismo estaría nominalmente activo y en la práctica muerto.

Por qué `-MaxPopulation 300`: la población se reescala a 4× el tamaño del
archivo. Sin tope, una sola generación podría costar más que el presupuesto
entero.

### Seguimiento

```powershell
Get-Content runs_weap\eps_iter02\seed42.log -Tail 5 -Wait
```

El log trae, en cada checkpoint: evaluaciones, hipervolumen, tamaño del archivo,
tamaño de la población y minutos transcurridos. Los reinicios aparecen como
`EpsNSGAII restarting; adjusting population size from N to M`.

### Reanudación

Automática. El `.ckpt` guarda el archivo ε, la población, el caché de
diagnóstico de J1/J6 y la historia de HV cada `--checkpoint_every` evaluaciones
(200 por defecto). Si el proceso muere, volver a lanzar continúa desde ahí. Si
el `.dat` final ya existe, esa semilla se salta.

Al reanudar se reinyecta el **archivo**, no la población: el archivo es el
estado valioso, y es además lo que el propio algoritmo usa al reiniciar
(`restart()` hace `population = archive[:] + mutantes`), de modo que reanudar
así es consistente con su semántica.

La reanudación **no reproduce bit a bit** una corrida sin interrupciones: no se
restaura el estado del generador aleatorio. Es estadísticamente equivalente,
pero una corrida reanudada no debe compararse con otra semilla como si fueran
réplicas exactas del mismo procedimiento.

---

## 7. Cómo leer los resultados

```powershell
.\venv_DPS\Scripts\python.exe weap_dps\comparar_algoritmos.py --eps "runs_weap\eps_iter02\pareto_seed*.dat"
```

Imprime tres cosas, que responden preguntas distintas y conviene no mezclar:

**Tabla por semilla.** HV, evaluaciones, tamaño del frente, horas, y
**fracción no dominada de la población** (columna `nd pobl.`). Es el
diagnóstico de presión de selección: en 1,00 el rango de Pareto no discrimina
nada dentro de la población, y lo único que decide quién sobrevive es el
apiñamiento.

Se calcula sobre la **población** y no sobre el frente, por una razón que una
versión anterior de esta guía pasaba por alto: el archivo ε contiene solo
soluciones mutuamente no dominadas **por construcción**, así que su fracción no
dominada vale siempre 1,00 y no diagnostica nada. En NSGA-II el frente reportado
es la población final, de modo que para NSGA-II las dos cosas coinciden.

**Solo es comparable a igual número de evaluaciones.** Al comienzo de cualquier
algoritmo genético la población tiene más soluciones dominadas, así que la cifra
parte baja y sube a medida que la búsqueda converge. La referencia de NSGA-II
—1,00 en las cinco semillas— está medida a las 4.000; la de ε-NSGA-II hay que
leerla también a las 4.000. A las 800 evaluaciones daba entre 0,74 y 0,97, un
valor que todavía no dice nada comparado con NSGA-II.

> **Hay que guardar una copia de los checkpoints a las 4.000.** El `.ckpt` se
> sobrescribe cada 200 evaluaciones y se borra al terminar, y el `.dat` final de
> la corrida lanzada el 20-09-2026 no trae la población —se agregó después de
> lanzarla—. Tanto la fracción no dominada de la población como el HV de la
> unión a las 4.000 existen solo en ese `.ckpt`, durante unas 3 horas. Cuando el
> log marque entre 4.000 y 4.200 evaluaciones (hacia el martes 23 entre las
> 15:30 y las 18:30):
>
> ```powershell
> New-Item -ItemType Directory -Force runs_weap\eps_iter02\snap_4000 | Out-Null; Copy-Item runs_weap\eps_iter02\*.ckpt runs_weap\eps_iter02\snap_4000\
> ```
>
> y la comparación a presupuesto igual se hace sobre esa copia:
>
> ```powershell
> python weap_dps\comparar_algoritmos.py --eps "runs_weap\eps_iter02\snap_4000\*.ckpt"
> ```
>
> El HV por semilla a las 4.000 también queda en `hv_history` y se recupera
> después, pero el HV de la **unión** y la población no.

**Curva HV(nfe).** Si sigue subiendo al agotarse el presupuesto, el presupuesto
fue corto. Se reporta la ganancia del último cuarto del presupuesto: si es
despreciable, la curva se aplanó.

**HV de la unión.** Mayor es mejor. Una diferencia del orden del CV entre
semillas (0,7 %) **no es evidencia de nada**.

---

## 8. Limitaciones que hay que declarar

**El presupuesto no es exacto.** La descendencia de un reinicio se evalúa en
bloque —`restart()` llama a `evaluate_all(offspring)` dentro de un paso—, así
que una corrida puede exceder lo pedido por hasta
`(max_population − tamaño del archivo)` evaluaciones. En la prueba de humo, con
ventana de 1 generación, 60 pedidas terminaron en 115.

No se "arregla" recortando el reinicio, porque mutilarlo cambiaría el algoritmo
que se quiere medir. Lo que se hace es registrar `nfe_real` en el `.dat` y en el
log, y **comparar contra NSGA-II por la curva HV(nfe)** —que permite leer ambos
al mismo número de evaluaciones— en vez de por el valor final de cada `.dat`,
que estaría medido con presupuestos distintos.

**El hipervolumen es una estimación, no el valor exacto.** El hipervolumen
exacto de Platypus es el algoritmo recursivo en Python puro y con 5 objetivos
escala de forma prohibitiva. Medido sobre nuestro propio frente:

| soluciones | tiempo |
|---|---|
| 10 | 0,00 s |
| 20 | 0,02 s |
| 40 | 0,54 s |
| 100 | ~40 s |
| 500 | no terminó en 25 min |

Como el archivo ε puede tener cientos de soluciones y el HV se calcula muchas
veces durante una corrida, el método exacto haría que **medir la convergencia
costara más que optimizar**. Se usa en cambio una estimación Monte Carlo
(`hv_utils.py`): queda a 0,15–0,36 % del exacto donde éste sí es calculable, y
tarda 0,6 s con 100 soluciones.

Con semilla y número de muestras fijos, dos conjuntos se evalúan contra los
mismos puntos de muestreo, así que el error queda correlacionado entre ellos y
las **comparaciones** son más precisas que el error absoluto de cada valor por
separado. Es exactamente el régimen que interesa. Aun así: al reportar, decir
que es una estimación.

**La caja del hipervolumen es fija y eso es deliberado.** El HV solo es
comparable entre checkpoints, semillas y algoritmos si la caja de normalización
no cambia; si se derivara del frente de cada corrida, cada corrida se mediría
contra su propia vara. Las soluciones que caen más allá del nadir se **recortan**
al nadir y se cuentan: si el contador `fuera` deja de ser cero, la caja quedó
chica y hay que ampliar `HV_MAXIMUM` en `config_weap.py` **y recalcular todo**,
no solo la corrida nueva.

**J1 y J6 siguen fuera del conjunto optimizado.** Con ε el argumento para
excluirlos es más limpio, pero reincorporarlos exigiría derivar sus ε y rehacer
la comparación. Queda para una iteración posterior.

---

## 9. Qué no responde este experimento

El optimizador **no es el eslabón débil de este trabajo**. Lo es la brecha entre
el desempeño del emulador dentro y fuera de distribución: el ajuste en el
reparto del suministro entre fuentes cae de 0,679 sobre el ensamble base a
0,439 sobre las corridas del frente. Esa diferencia es de un orden de magnitud
mayor que cualquier diferencia esperable entre MOEAs.

Este experimento sirve para poder declarar, con evidencia y no por omisión, que
la elección del algoritmo y el tamaño del presupuesto fueron examinados. No
sustituye a la verificación del emulador contra WMS2Ma.
