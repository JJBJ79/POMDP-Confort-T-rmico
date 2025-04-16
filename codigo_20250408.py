
import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy
import time
global max_part
max_part = 100000

class AgentePersonalizado(pomdp_py.Agent):
    def __init__(self, belief, policy, transition_model, observation_model, reward_model):
        try:        
            assert isinstance(policy, ModeloDePolitica), "[ERROR] policy no es una instancia de ModeloDePolitica"

            self.policy = policy

            super().__init__(belief, self.policy, transition_model, observation_model, reward_model)

            if self.policy is None:
                raise AttributeError("[ERROR] 'policy' no fue asignado correctamente en la clase base.")
            
            print(f"[DEBUG] Política asignada correctamente: {self.policy}")

        except Exception as e:
            print(f"[ERROR] Falló la inicialización de AgentePersonalizado: {e}")
            raise e

        self.mi_policy = policy
        self.mi_belief = belief
        self.mi_transition_model = transition_model
        self.mi_observation_model = observation_model
        self.mi_reward_model = reward_model

        print("[DEBUG] Agente personalizado creado:")
        print(f"Belief: {self.mi_belief}")
        print(f"Policy: {self.mi_policy}")
        print(f"Transition Model: {self.mi_transition_model}")
        print(f"Observation Model: {self.mi_observation_model}")
        print(f"Reward Model: {self.mi_reward_model}")

class EstadoConfortTermico(pomdp_py.State):
    def __init__(self, ambiente, control, persona, transferencia):
        super().__init__()  
        self.ambiente = ambiente
        self.control = control
        self.persona = persona
        self.transferencia = transferencia

    def __hash__(self):
        return hash((
            tuple(sorted(self.ambiente.items())),
            tuple(sorted(self.control.items())),
            tuple(sorted(self.persona.items())),
            tuple(sorted(self.transferencia.items()))
        ))

    def __eq__(self, other):
        if isinstance(other, EstadoConfortTermico):
            return (self.ambiente == other.ambiente and
                    self.control == other.control and
                    self.persona == other.persona and
                    self.transferencia == other.transferencia)
        return False

    def __repr__(self):
        return (f"EstadoConfortTermico(ambiente={self.ambiente}, control={self.control}, "
                f"persona={self.persona}, transferencia={self.transferencia})")

class Particula(pomdp_py.State):
    def __init__(self, estado, weight=1.0):
        super().__init__()
        self.estado = estado
        self.weight = weight
        self.ambiente = estado.ambiente
        self.control = estado.control
        self.persona = estado.persona
        self.transferencia = estado.transferencia

    def __repr__(self):
        return f"Particula(estado={self.estado}, weight={self.weight})"

    def __eq__(self, other):
        if isinstance(other, Particula):
            return self.estado == other.estado and self.weight == other.weight
        return False

    def __hash__(self):
        return hash((self.estado, self.weight))

class AgentePersonalizado(pomdp_py.Agent):
    def __init__(self, belief, policy, transition_model, observation_model, reward_model):
        try:          
            assert isinstance(policy, ModeloDePolitica), "[ERROR] policy no es una instancia de ModeloDePolitica"
            self.policy = policy

            super().__init__(belief, self.policy, transition_model, observation_model, reward_model)

            if self.policy is None:
                raise AttributeError("[ERROR] 'policy' no fue asignado correctamente en la clase base.")
            
            print(f"[DEBUG] Política asignada correctamente: {self.policy}")

        except Exception as e:
            print(f"[ERROR] Falló la inicialización de AgentePersonalizado: {e}")
            raise e

        self.mi_policy = policy
        self.mi_belief = belief
        self.mi_transition_model = transition_model
        self.mi_observation_model = observation_model
        self.mi_reward_model = reward_model

        print("[DEBUG] Agente personalizado creado:")
        print(f"Belief: {self.mi_belief}")
        print(f"Policy: {self.mi_policy}")
        print(f"Transition Model: {self.mi_transition_model}")
        print(f"Observation Model: {self.mi_observation_model}")
        print(f"Reward Model: {self.mi_reward_model}")

class AccionConfortTermico(pomdp_py.Action):
    def __init__(self, cambio_de_temperatura, cambio_de_flujo_de_aire=0):
        self.cambio_de_temperatura = cambio_de_temperatura
        self.cambio_de_flujo_de_aire = cambio_de_flujo_de_aire

    def __repr__(self):
        return (f"AccionConfortTermico(cambio_de_temperatura={self.cambio_de_temperatura}, "
                f"cambio_de_flujo_de_aire={self.cambio_de_flujo_de_aire})")

    def __eq__(self, other):
        if isinstance(other, AccionConfortTermico):
            return (self.cambio_de_temperatura == other.cambio_de_temperatura and
                    self.cambio_de_flujo_de_aire == other.cambio_de_flujo_de_aire)
        return False

    def __hash__(self):
        return hash((self.cambio_de_temperatura, self.cambio_de_flujo_de_aire))

    def ejecutar_accion(self):
        """Método de depuración para mostrar qué acción se está ejecutando."""
        print(f"[DEBUG] Ejecutando acción: "
              f"temperatura={self.cambio_de_temperatura}, flujo_de_aire={self.cambio_de_flujo_de_aire}")

    def es_accion_valida(self):
        """Método adicional para verificar si los valores son válidos."""
        if not (-1 <= self.cambio_de_temperatura <= 1):
            print(f"[ERROR] Valor de cambio_de_temperatura inválido: {self.cambio_de_temperatura}")
            return False
        if not (-1 <= self.cambio_de_flujo_de_aire <= 1):
            print(f"[ERROR] Valor de cambio_de_flujo_de_aire inválido: {self.cambio_de_flujo_de_aire}")
            return False
        print("[DEBUG] La acción es válida.")
        return True

class ObservacionConfort(pomdp_py.Observation):
    def __init__(self, Tt, Ta, Var, Hr):
        self.Tt = Tt
        self.Ta = Ta
        self.Var = Var
        self.Hr = Hr

    def __repr__(self):
        return (f"ObservacionConfort(Tt={self.Tt}, Ta={self.Ta}, "
                f"Var={self.Var}, Hr={self.Hr})")

    def __eq__(self, other):
        if isinstance(other, ObservacionConfort):
            return (self.Tt == other.Tt and self.Ta == other.Ta and
                    self.Var == other.Var and self.Hr == other.Hr)
        return False

    def __hash__(self):
        return hash((self.Tt, self.Ta, self.Var, self.Hr))


class ModeloDeObservacion(pomdp_py.ObservationModel):
    def __init__(self, ruido_observacion=0.15):
        self.ruido_observacion = ruido_observacion

    def sample(self, estado, accion):
        if not hasattr(estado, "control") or not hasattr(estado, "ambiente"):
            raise ValueError("[ERROR] El estado no contiene las claves 'control' o 'ambiente'.")

        Tt_valor = max(16, min(32, estado.control.get("Tt", 0) + random.uniform(-self.ruido_observacion, self.ruido_observacion)))
        Ta_valor = max(5, min(35, estado.ambiente.get("Ta", 0) + random.uniform(-self.ruido_observacion, self.ruido_observacion)))
        Var_valor = max(0.1, min(1.0, estado.control.get("Var", 0) + random.uniform(-self.ruido_observacion, self.ruido_observacion)))
        Hr_valor = max(10, min(70, estado.ambiente.get("Hr", 0) + random.uniform(-self.ruido_observacion, self.ruido_observacion)))

        return ObservacionConfort(Tt=Tt_valor, Ta=Ta_valor, Var=Var_valor, Hr=Hr_valor)

    def probability(self, obs, estado, accion):
        probabilidad = 1.0
        for atributo, valor_obs in vars(obs).items():
            if atributo in estado.ambiente:
                valor_estado = estado.ambiente.get(atributo, 0)
            elif atributo in estado.control:
                valor_estado = estado.control.get(atributo, 0)
            else:
                continue

            sigma = 5
            prob_atributo = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((valor_obs - valor_estado) / sigma)**2)
            probabilidad *= prob_atributo

        return probabilidad

    def get_all_observations(self):
        observaciones = []
        for Tt in range(16, 32, 2):
            for Ta in range(5, 35, 2):
                for Var in [0.1, 0.3, 0.5, 0.7, 1.0]:
                    for Hr in range(10, 70, 10):
                        observaciones.append(ObservacionConfort(Tt=Tt, Ta=Ta, Var=Var, Hr=Hr))
        return observaciones

class ModeloDeTransicion(pomdp_py.TransitionModel):
    def __init__(self, estados=None):
        super().__init__()
        self.estados = estados if estados else []

    def get_all_states(self, max_part=100000):
        if not self.estados:
            self.estados = [
                EstadoConfortTermico(
                    ambiente={
                        "Hr": round(random.uniform(30, 70), 1),
                        "Pa": round(random.uniform(90, 110), 1),
                        "Ta": round(random.uniform(15, 30), 1),
                        "Tr": round(random.uniform(18, 28), 1),
                    },
                    control={
                        "Tt": round(random.uniform(20, 26), 1),
                        "Var": round(random.uniform(0.1, 1.0), 2),
                    },
                    persona={
                        "M": round(random.uniform(0.8, 1.5), 2),
                        "W": round(random.uniform(0, 0.2), 2),
                        "Icl": round(random.uniform(0.5, 2.0), 2),
                    },
                    transferencia={
                        "hc": round(random.uniform(0.3, 1.0), 2),
                        "Tcl": round(random.uniform(22, 32), 1),
                    },
                )
                for _ in range(max_part)
            ]
        return self.estados

    def probability(self, estado_siguiente, estado_actual, accion):
        sigmas_ambiente = [1.0, 0.5, 0.1]
        sigmas_control = [0.5, 0.2, 0.1]
        sigmas_persona = [0.3, 0.1]
        sigmas_transferencia = [0.2, 0.1]

        prob_A = self._compute_probability(
            estado_actual.ambiente["Ta"], estado_siguiente.ambiente["Ta"], sigmas_ambiente, delta=accion.cambio_de_temperatura)
        prob_C = self._compute_probability(
            estado_actual.control["Tt"], estado_siguiente.control["Tt"], sigmas_control, delta=accion.cambio_de_temperatura)
        prob_F = self._compute_probability(
            estado_actual.control["Var"], estado_siguiente.control["Var"], sigmas_control, delta=accion.cambio_de_flujo_de_aire)
        prob_P = self._compute_probability(
            estado_actual.persona["M"], estado_siguiente.persona["M"], sigmas_persona)
        prob_T = self._compute_probability(
            estado_actual.transferencia["hc"], estado_siguiente.transferencia["hc"], sigmas_transferencia)

        return prob_A * prob_C * prob_F * prob_P * prob_T

    def sample(self, estado_actual, accion):
        ambiente = estado_actual.ambiente.copy()
        control = estado_actual.control.copy()
        persona = estado_actual.persona.copy()
        transferencia = estado_actual.transferencia.copy()

        # Ajuste de temperatura
        if accion.cambio_de_temperatura == 1:
            ambiente["Ta"] = min(30, ambiente["Ta"] + 1)
        elif accion.cambio_de_temperatura == -1:
            ambiente["Ta"] = max(15, ambiente["Ta"] - 1)

        # Ajuste de flujo de aire
        if accion.cambio_de_flujo_de_aire == 1:
            control["Var"] = min(1.0, control["Var"] + 0.1)
            print("[DEBUG] Incrementando flujo de aire a:", control["Var"])
        elif accion.cambio_de_flujo_de_aire == -1:
            control["Var"] = max(0.1, control["Var"] - 0.1)
            print("[DEBUG] Reduciendo flujo de aire a:", control["Var"])
        else:
            print("[DEBUG] Manteniendo flujo de aire en:", control["Var"])

        return EstadoConfortTermico(ambiente=ambiente, control=control, persona=persona, transferencia=transferencia)

    def _compute_probability(self, valor_actual, valor_siguiente, sigmas, factor=1.0, delta=0.0):
        sigma = sigmas[0] if len(sigmas) > 0 else 1.0
        mu = valor_actual + delta
        probabilidad = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((valor_siguiente - mu) / sigma) ** 2)
        return probabilidad * factor

class ModeloDeRecompensa(pomdp_py.RewardModel):
    def _funcion_recompensa(self, estado, accion):
        try:
            Ta = estado.ambiente["Ta"]  
            Tt = estado.control["Tt"] 
            hc = estado.transferencia["hc"] 
        except KeyError as e:
            raise ValueError(f"Falta clave en estado: {e}")

        if accion.cambio_de_temperatura == 1:  
            if Ta < Tt: 
                return 10 
            else:
                return -10  
        elif accion.cambio_de_temperatura == -1:  
            if Ta > Tt:  
                return 10  
            else:
                return -10  
        elif accion.cambio_de_flujo_de_aire == 1:  
            if hc < 1.0:  
                return 5  
            else:
                return -5 
        elif accion.cambio_de_flujo_de_aire == -1: 
            if hc > 0.5:  
                return 5  
            else:
                return -5  
        else:
            return -1  

    def sample(self, estado, accion, estado_siguiente):
        return self._funcion_recompensa(estado, accion)
    
class ModeloDePolitica(pomdp_py.RolloutPolicy):
    def __init__(self):
        super().__init__()
        self.acciones = [
            AccionConfortTermico(cambio_de_temperatura=1, cambio_de_flujo_de_aire=0),
            AccionConfortTermico(cambio_de_temperatura=-1, cambio_de_flujo_de_aire=0),
            AccionConfortTermico(cambio_de_temperatura=0, cambio_de_flujo_de_aire=1),  
            AccionConfortTermico(cambio_de_temperatura=0, cambio_de_flujo_de_aire=-1), 
            AccionConfortTermico(cambio_de_temperatura=0, cambio_de_flujo_de_aire=0) 
        ]

    def sample(self, estado):
        accion_seleccionada = random.choice(self.acciones)
        print(f"[DEBUG] Acción seleccionada en sample: {accion_seleccionada}")
        return accion_seleccionada

    def rollout(self, estado, historial=None):

        accion_durante_rollout = self.sample(estado)
        print(f"[DEBUG] Acción utilizada durante rollout: {accion_durante_rollout}")
        return accion_durante_rollout

    def get_all_actions(self, state=None, history=None):

        print(f"[DEBUG] Acciones disponibles: {self.acciones}")
        return self.acciones
    
class EntornoConfort(pomdp_py.Environment):
    def __init__(self, estado_inicial, modelo_transicion, modelo_recompensas):
        super().__init__(estado_inicial, modelo_transicion, modelo_recompensas)
        self.estado_actual = estado_inicial  


class ProblemaConfortTermico(pomdp_py.POMDP):
    def __init__(self, agente, entorno):
        super().__init__(agente, entorno)
        self.agente = agente
        self.entorno = entorno  

    @staticmethod
    def crear(ruido_observacion):
        print("[DEBUG] Comenzando la creación del problema.")

        estado_inicial = EstadoConfortTermico(
            ambiente={"Hr": 38.0, "Pa": 93.9, "Ta": 24.1, "Tr": 25.3},
            control={"Tt": 20.2, "Var": 0.75},
            persona={"M": 1.05, "W": 0.18, "Icl": 1.86},
            transferencia={"hc": 0.92, "Tcl": 22.4}
        )

        creencia_inicial = crear_creencia_inicial_en_particulas(num_particulas=10000)

        policy_model = ModeloDePolitica()
        transition_model = ModeloDeTransicion()
        observation_model = ModeloDeObservacion(ruido_observacion)
        reward_model = ModeloDeRecompensa()

        try:
            agente = AgentePersonalizado(
                belief=creencia_inicial,
                policy=policy_model,
                transition_model=transition_model,
                observation_model=observation_model,
                reward_model=reward_model
            )
        except Exception as e:
            print(f"[ERROR] Falló la inicialización del agente: {e}")
            raise e

        entorno = EntornoConfort(estado_inicial, transition_model, reward_model)
        problema = ProblemaConfortTermico(agente, entorno)
        return problema

def crear_creencia_inicial_en_particulas(num_particulas=10000):
    estados_posibles = [
        EstadoConfortTermico(
            ambiente={"Hr": 38.0, "Pa": 93.9, "Ta": 24.1, "Tr": 25.3},
            control={"Tt": 20.2, "Var": 0.75},
            persona={"M": 1.05, "W": 0.18, "Icl": 1.86},
            transferencia={"hc": 0.92, "Tcl": 22.4}
        )
    ]

    particulas = [
        Particula(estado=random.choice(estados_posibles), weight=1.0 / num_particulas)
        for _ in range(num_particulas)
    ]

    return pomdp_py.Particles(particulas)

global historial_n_eff
if "historial_n_eff" not in globals():
    historial_n_eff = []

def update_belief_with_resample(agente, accion, observacion_real):
    global iteraciones_resampleo
    if "iteraciones_resampleo" not in globals():
        iteraciones_resampleo = 0  

    factor_actualizacion = 1
    peso_minimo = 0.8  

    for particula in agente.cur_belief.particles:
        particula.weight = max(particula.weight * factor_actualizacion, peso_minimo)  

    total_weight = sum(particula.weight for particula in agente.cur_belief.particles)
    if total_weight > 0:
        for particula in agente.cur_belief.particles:
            particula.weight /= total_weight
            particula.weight += np.random.uniform(-0.005, 0.005)  
            particula.weight = max(particula.weight, peso_minimo)  

    n_eff = 1.0 / (sum(particula.weight**2 for particula in agente.cur_belief.particles) + 1e-6)
    n_eff = max(n_eff, len(agente.cur_belief.particles) * 0.3)  # 🔹 Aseguramos un umbral mínimo de estabilidad
    historial_n_eff.append(n_eff)
    print(f"[DEBUG] Iteración {iteraciones_resampleo}: Número efectivo de partículas = {n_eff}")

    umbral_resampleo = 0.25  
    if n_eff < len(agente.cur_belief.particles) * umbral_resampleo:
        iteraciones_resampleo += 1  
        if iteraciones_resampleo > 5:  
            print("[ERROR] Se ha excedido el número de resampleos, deteniendo...")
            return  

        estados_posibles = ModeloDeTransicion().get_all_states()
        if not estados_posibles:
            print("[ERROR] No hay estados posibles para el resampleo.")
            return  

        particulas_importantes = sorted(agente.cur_belief.particles, key=lambda p: p.weight, reverse=True)[:500]
        nuevas_particulas = []

        for _ in range(10000):
            particula_base = random.choice(particulas_importantes)  
            estado_modificado = EstadoConfortTermico(
                ambiente={k: min(max(v + np.random.normal(0, 4.5), 15), 35) for k, v in particula_base.estado.ambiente.items()},
                control={k: min(max(v + np.random.normal(0, 1.2), 0.1), 1.0) for k, v in particula_base.estado.control.items()},
                persona={k: min(max(v + np.random.normal(0, 1.8), 0.8), 2.0) for k, v in particula_base.estado.persona.items()},
                transferencia={k: min(max(v + np.random.normal(0, 1.0), 0.3), 1.0) for k, v in particula_base.estado.transferencia.items()},
            )
            nuevas_particulas.append(Particula(estado_modificado, weight=np.random.uniform(0.8, 1.2)))
            
            nuevas_particulas = [p for p in nuevas_particulas if p.estado.Ta > 18 and p.estado.Tr > 18]

        total_weight = sum(particula.weight for particula in nuevas_particulas)
        if total_weight > 0:
            for particula in nuevas_particulas:
                particula.weight = max(particula.weight / total_weight, np.random.uniform(1.0, 2.5))  
                particula.weight += np.random.uniform(0.1, 0.3)  

            promedio_peso = np.mean([p.weight for p in nuevas_particulas])  
            for particula in nuevas_particulas:
                particula.weight = max(particula.weight, promedio_peso * 0.5)  

        agente.set_belief(pomdp_py.Particles(nuevas_particulas))


        print("[INFO] Historial de número efectivo de partículas:", historial_n_eff)

def probar_planificador(problema_confort, planificador, pasos=3):
    for i in range(pasos):
        print(f"\n[DEBUG] === Paso {i + 1} ===")
        accion = planificador.plan(problema_confort.agente)
        print(f"[DEBUG] Acción seleccionada: {accion}")
        observacion_real = problema_confort.agente.observation_model.sample(
            problema_confort.entorno.estado_actual, accion
        )
        print(f"[DEBUG] Observación generada: {observacion_real}")

        update_belief_with_resample(problema_confort.agente, accion, observacion_real)
        planificador.update(problema_confort.agente, accion, observacion_real)
        print(f"[DEBUG] Finalizado el paso {i + 1}")
        print(f"[DEBUG] Comenzando ejecución del método plan() con el agente.")

def update_belief_with_resample(agente, accion, observacion_real):
    global iteraciones_resampleo

    if "iteraciones_resampleo" not in globals():
        iteraciones_resampleo = 0  

    factor_actualizacion = 1
    peso_minimo = 0.5

    for particula in agente.cur_belief.particles:
        particula.weight = max(particula.weight * factor_actualizacion, peso_minimo)  

    total_weight = sum(particula.weight for particula in agente.cur_belief.particles)
    if total_weight > 0:
        for particula in agente.cur_belief.particles:
            particula.weight /= total_weight
            particula.weight += np.random.uniform(-0.005, 0.005)  
            particula.weight = max(particula.weight, peso_minimo)  
            print(f"[DEBUG] Estado: {particula.estado}, Peso: {particula.weight}")

    n_eff = 1.0 / (sum(particula.weight**2 for particula in agente.cur_belief.particles) + 1e-6)
    print(f"[DEBUG] Número efectivo de partículas: {n_eff}")

    umbral_resampleo = 0.35  
    if n_eff < len(agente.cur_belief.particles) * umbral_resampleo:
        iteraciones_resampleo += 1  
        if iteraciones_resampleo > 5:  
            print("[ERROR] Se ha excedido el número de resampleos, deteniendo...")
            return  

        print("[DEBUG] Número de partículas demasiado bajo, aplicando resampleo ")
        estados_posibles = ModeloDeTransicion().get_all_states()

        if not estados_posibles:
            print("[ERROR] No hay estados posibles para el resampleo.")
            return  

        particulas_importantes = sorted(agente.cur_belief.particles, key=lambda p: p.weight, reverse=True)[:500]
        
        # 🔹 Generamos nuevas partículas con diversidad
        nuevas_particulas = []
        for _ in range(10000):  
            particula_base = random.choice(particulas_importantes)  
            estado_modificado = EstadoConfortTermico(
                ambiente={k: min(max(v + np.random.normal(0, 5.0), 15), 35) for k, v in particula_base.estado.ambiente.items()},
                control={k: min(max(v + np.random.normal(0, 1.5), 0.1), 1.0) for k, v in particula_base.estado.control.items()},
                persona={k: min(max(v + np.random.normal(0, 2.0), 0.8), 2.0) for k, v in particula_base.estado.persona.items()},
                transferencia={k: min(max(v + np.random.normal(0, 1.5), 0.3), 1.0) for k, v in particula_base.estado.transferencia.items()},
            )
            nuevas_particulas.append(Particula(estado_modificado, weight=np.random.uniform(0.8, 2.5)))    

        for i in range(min(5, len(nuevas_particulas))):  
            print(f"[DEBUG] Estado de Partícula {i}: {nuevas_particulas[i].estado}")

        if not nuevas_particulas:
            print("[ERROR] La lista de nuevas partículas está vacía después del resampleo.")
            return  

        print(f"[DEBUG] Número de partículas después del resampleo: {len(nuevas_particulas)}")

        total_weight = sum(particula.weight for particula in nuevas_particulas)
        if total_weight > 0:
            for particula in nuevas_particulas:
                particula.weight = max(particula.weight / total_weight, np.random.uniform(0.8, 1))  
                particula.weight += np.random.uniform(0.1, 0.3)  

            promedio_peso = np.mean([p.weight for p in nuevas_particulas])  
            for particula in nuevas_particulas:
                particula.weight = max(particula.weight, promedio_peso * 0.5)  

            for i in range(min(10, len(nuevas_particulas))):  
                print(f"[DEBUG] Peso de Partícula {i}: {nuevas_particulas[i].weight}")              

        agente.set_belief(pomdp_py.Particles(nuevas_particulas))


def prueba_basica():
    print("[DEBUG] Ejecutando prueba básica.")

    estado_inicial = EstadoConfortTermico(
        ambiente={"Hr": 38.0, "Pa": 93.9, "Ta": 24.1, "Tr": 25.3},
        control={"Tt": 20.2, "Var": 0.75},
        persona={"M": 1.05, "W": 0.18, "Icl": 1.86},
        transferencia={"hc": 0.92, "Tcl": 22.4},
    )
    print("[DEBUG] Estado inicial creado:", estado_inicial)

    modelo_transicion = ModeloDeTransicion()
    print("[DEBUG] Modelo de Transición creado:", modelo_transicion)

    creencia_inicial = crear_creencia_inicial_en_particulas(5)  
    print("[DEBUG] Creencia inicial creada:", creencia_inicial)

    pouct = pomdp_py.POUCT(
        max_depth=10,  
        discount_factor=0.95,
        num_sims=5000, 
        exploration_const=10,
        rollout_policy=None,  
    )
    print("[DEBUG] POUCT configurado:", pouct)

    try:
        accion = pouct.plan(None)  
        print(f"[DEBUG] Acción seleccionada por POUCT: {accion}")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error al ejecutar POUCT: {e}")

def main():
    print("[DEBUG] La función 'main' ha comenzado.")

    confort = ProblemaConfortTermico.crear(ruido_observacion=0.15)
    print(f"[DEBUG] ProblemaConfortTermico creado: {confort}")
     
    if not isinstance(confort, ProblemaConfortTermico):
        print("[ERROR] `confort` no es una instancia válida de `ProblemaConfortTermico`. Verifica la función `crear()`.")
        return
    
    print(f"[DEBUG] Agente en ProblemaConfortTermico: {confort.agente}")
    print(f"[DEBUG] Modelo de Transición: {confort.agente.transition_model}")
    print(f"[DEBUG] Creencia inicial: {confort.agente.belief}")

    try:
        print("[DEBUG] Verificando política del agente antes de inicializar POUCT...")
        
        if not hasattr(confort.agente, "policy") or confort.agente.policy is None:
            print("[ERROR] La política del agente no está inicializada. Se usará una política por defecto.")
            confort.agente.policy = ModeloDePolitica()  
        
        if not hasattr(confort.agente.policy, "get_all_actions"):
            print("[ERROR] La política del agente no tiene el método 'get_all_actions'.")
            return
        
        if not hasattr(confort.agente.policy, "rollout"):
            print("[ERROR] La política del agente no tiene el método 'rollout'. Se asignará un método de respaldo.")
            setattr(confort.agente.policy, "rollout", lambda s: confort.agente.policy.get_all_actions()[0])  

        print("[DEBUG] Acciones disponibles en la política:", confort.agente.policy.get_all_actions())
        print("[DEBUG] Acción de prueba (rollout):", confort.agente.policy.rollout(None))

    except Exception as e:
        print(f"[ERROR] Ocurrió un error inesperado: {e}")
        return

    if confort.agente.policy is None:
        confort.agente.policy = ModeloDePolitica()
    pouct = pomdp_py.POUCT(
        max_depth=10,
        discount_factor=0.95,
        num_sims=5000,
        exploration_const=0.8,
        rollout_policy=confort.agente.policy
    )

    print("\n** Pruebas de POUCT con cambio de flujo de aire **")
    try:
        pouct = pomdp_py.POUCT(
            max_depth=50,
            discount_factor=0.95,
            num_sims=5000,
            exploration_const=1.2,
            rollout_policy=confort.agente.policy, 
            show_progress=True,
        )
        probar_planificador(confort, pouct, pasos=3)  
    except Exception as e:
        print(f"[ERROR] Ocurrió un error durante la configuración o ejecución de POUCT: {e}")
    
    print("\n** Pruebas de POMCP con cambio de flujo de aire **")
    try:
        pomcp = pomdp_py.POMCP(
            max_depth=50,
            discount_factor=0.95,
            num_sims=5000,  
            exploration_const=1.2,
            rollout_policy=confort.agente.policy,
            show_progress=True,
            pbar_update_interval=500,
        )
        probar_planificador(confort, pomcp, pasos=3)  
    except Exception as e:
        print(f"[ERROR] Ocurrió un error durante la configuración o ejecución de POMCP: {e}")

    print("[DEBUG] La función 'main' ha terminado.")

if __name__ == "__main__":
    try:
        print("[DEBUG] Iniciando prueba básica antes de la función main.")
        prueba_basica() 
        main()  
    except Exception as e:
        print(f"[ERROR] Ocurrió un error durante la ejecución global: {e}")
    finally:
        print("[INFO] Ejecución del programa finalizada.")
