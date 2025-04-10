import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import numpy as np
import sys
import copy
import time
global max_part
max_part = 500

class AgentePersonalizado(pomdp_py.Agent):
    def __init__(self, belief, policy, transition_model, observation_model, reward_model):
        
        super().__init__(belief, policy, transition_model, observation_model, reward_model)

        self.belief = belief
        self.policy = policy
        self.transition_model = transition_model
        self.observation_model = observation_model
        self.reward_model = reward_model

        print(f"[DEBUG] Agente personalizado creado:")
        print(f"Belief: {belief}")
        print(f"Policy: {policy}")
        print(f"Transition Model: {transition_model}")
        print(f"Observation Model: {observation_model}")
        print(f"Reward Model: {reward_model}")

class Particula:
    def __init__(self, estado, weight=1.0):
        
        self.estado = estado
        self.weight = weight

    def __repr__(self):
        return f"Particula(estado={self.estado}, weight={self.weight})"

class EstadoConfortTermico(pomdp_py.State):
    def __init__(self, ambiente, control, persona, transferencia):
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

    def get_all_states(self, max_part=100):
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
            estado_actual.ambiente["Ta"], estado_siguiente.ambiente["Ta"], sigmas_ambiente, delta=accion.cambio_de_temperatura
        )
        prob_C = self._compute_probability(
            estado_actual.control["Tt"], estado_siguiente.control["Tt"], sigmas_control, delta=accion.cambio_de_temperatura
        )
        prob_P = self._compute_probability(
            estado_actual.persona["M"], estado_siguiente.persona["M"], sigmas_persona
        )
        prob_T = self._compute_probability(
            estado_actual.transferencia["hc"], estado_siguiente.transferencia["hc"], sigmas_transferencia
        )

        return prob_A * prob_C * prob_P * prob_T

    def sample(self, estado_actual, accion):

        ambiente = estado_actual.ambiente.copy()
        control = estado_actual.control.copy()
        persona = estado_actual.persona.copy()
        transferencia = estado_actual.transferencia.copy()

        if accion.cambio_de_temperatura == 1:
            ambiente["Ta"] = min(30, ambiente["Ta"] + 1) 
        elif accion.cambio_de_temperatura == -1:
            ambiente["Ta"] = max(15, ambiente["Ta"] - 1)  

        if accion.cambio_de_flujo_de_aire == 1:
            control["Var"] = min(1.0, control["Var"] + 0.1) 
        elif accion.cambio_de_flujo_de_aire == -1:
            control["Var"] = max(0.1, control["Var"] - 0.1) 

        return EstadoConfortTermico(ambiente=ambiente, control=control, persona=persona, transferencia=transferencia)

    def _compute_probability(self, valor_actual, valor_siguiente, sigmas, factor=1.0, delta=0.0):
        
        sigma = sigmas[0] if len(sigmas) > 0 else 1.0  
        mu = valor_actual + delta 
        probabilidad = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((valor_siguiente - mu) / sigma)**2)
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
            AccionConfortTermico(cambio_de_temperatura=1),
            AccionConfortTermico(cambio_de_temperatura=-1),
            AccionConfortTermico(cambio_de_temperatura=0),
        ]

    def sample(self, estado):
 
        return random.choice(self.acciones)

    def rollout(self, estado, historial=None):
       return self.sample(estado)

    def get_all_actions(self, state=None, history=None):
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

        creencia_inicial = crear_creencia_inicial_en_particulas(num_particulas=500)

        policy_model = ModeloDePolitica()
        transition_model = ModeloDeTransicion()
        observation_model = ModeloDeObservacion(ruido_observacion)
        reward_model = ModeloDeRecompensa()

        print(f"[DEBUG] Creencia inicial: {type(creencia_inicial)} - {creencia_inicial}")
        print(f"[DEBUG] Policy Model: {policy_model}")
        print(f"[DEBUG] Transition Model: {transition_model}")
        print(f"[DEBUG] Observation Model: {observation_model}")
        print(f"[DEBUG] Reward Model: {reward_model}")

        try:
            agente = AgentePersonalizado(
                belief=creencia_inicial,
                policy=policy_model,
                transition_model=transition_model,
                observation_model=observation_model,
                reward_model=reward_model
            )
            print(f"[DEBUG] Agente creado: {agente}")
        except Exception as e:
            print(f"[ERROR] Falló la inicialización del agente personalizado: {e}")
            raise e

        entorno = EntornoConfort(estado_inicial, transition_model, reward_model)
        print(f"[DEBUG] Entorno creado: {entorno}")

        problema = ProblemaConfortTermico(agente, entorno)
        print("[DEBUG] Problema creado exitosamente.")
        return problema

def crear_creencia_inicial_en_particulas(num_particulas=500):
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

def update_belief_with_resample(agente, accion, observacion_real):
    # Actualizar pesos de las partículas
    for particula in agente.cur_belief.particles:
        particula.weight *= agente.observation_model.probability(observacion_real, particula.estado, accion)

    total_weight = sum(particula.weight for particula in agente.cur_belief.particles)
    if total_weight > 0:
        for particula in agente.cur_belief.particles:
            particula.weight /= total_weight
            print(f"[DEBUG] Estado: {particula.estado}, Peso: {particula.weight}")

    # Calcular número efectivo de partículas (N_eff)
    n_eff = 1 / sum(particula.weight**2 for particula in agente.cur_belief.particles)
    print(f"[DEBUG] Número efectivo de partículas: {n_eff}")

    umbral_resampleo = 0.5
    if n_eff < len(agente.cur_belief.particles) * umbral_resampleo:
        print("[DEBUG] Resampleando partículas para evitar deprivación...")
        estados_posibles = ModeloDeTransicion().get_all_states()
        if estados_posibles:
            agente.cur_belief.particles = [
                Particula(random.choice(estados_posibles), weight=1.0 / len(estados_posibles))
                for _ in range(500)
            ]
        else:
            print("[ERROR] No hay estados posibles para realizar el re-sampling.")

    agente.update_history(accion, observacion_real)

def main():
    print("[DEBUG] La función 'main' ha comenzado.")
    confort = ProblemaConfortTermico.crear(ruido_observacion=0.15)
    print(f"[DEBUG] ProblemaConfortTermico creado: {confort}")
    print(f"[DEBUG] Agente en ProblemaConfortTermico: {confort.agente}")

    print("\n** Pruebas de POUCT **")
    pouct = pomdp_py.POUCT(
        max_depth=200,
        discount_factor=0.95,
        num_sims=50,
        exploration_const=50,
        rollout_policy=confort.agente.policy,
        show_progress=True,
    )
    probar_planificador(confort, pouct, pasos=3)

    print("\n** Pruebas de POMCP **")
    pomcp = pomdp_py.POMCP(
        max_depth=200,
        discount_factor=0.95,
        num_sims=200,
        exploration_const=50,
        rollout_policy=confort.agente.policy,
        show_progress=True,
        pbar_update_interval=500,
    )
    probar_planificador(confort, pomcp, pasos=3)

    print("[DEBUG] La función 'main' ha terminado.")
    
    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            print(f"[ERROR] Ocurrió un error durante la ejecución: {e}")