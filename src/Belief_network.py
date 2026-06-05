from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#import matplotlib.pyplot as plt # Per la visualizzazione del grafo (opzionale)
#import networkx as nx # Per la visualizzazione del grafo (opzionale)

class UncertaintyModule:
    def __init__(self, kg_module, component_name):
        self.kg = kg_module  # Colleghiamo il Knowledge Graph
        self.component = component_name 
        self.model = DiscreteBayesianNetwork([('load', 'ML_Output'), ('real_failure', 'ML_Output')])
        self._setup_cpds()
        self.inference = VariableElimination(self.model)

    def _setup_cpds(self):

        # 2. INTERROGAZIONE DEL KG
        # Recuperiamo il contesto dal KG (se è un pezzo fisico, se è in una zona critica, ecc.)
        context = self.kg.get_task_context(self.component)
        
        # Logica: Se il componente è critico (es. fa parte della Unit 2),
        # la probabilità a priori di guasto reale è più alta (es. 20% invece di 10%)
        if context and context.get('unit') == "unit_2":
            prob_guasto_true = 0.20
        else:
            prob_guasto_true = 0.10
            
        prob_guasto_false = 1.0 - prob_guasto_true

        # Probabilità a priori: 70% carico normale, 30% carico alto
        cpd_carico = TabularCPD(variable='load', variable_card=2, values=[[0.7], [0.3]])
        
        # Usiamo le variabili estratte dal KG 
        cpd_guasto = TabularCPD(
            variable='real_failure',
            variable_card=2,
            values=[[prob_guasto_false], [prob_guasto_true]]
        )

        # Tabella di condizionamento: riflette l'accuratezza del modello ML (es. Random Forest)
        # Se c'è un guasto e il carico è normale, il ML ha il 99% di probabilità di vederlo.
        # Se NON c'è un guasto ma il carico è alto, il ML potrebbe dare un falso allarme (60%).
        cpd_ml = TabularCPD(
            variable='ML_Output', 
            variable_card=2,
            values=[
                [0.95, 0.40, 0.05, 0.01], # no failure probability
                [0.05, 0.60, 0.95, 0.99]  # failure probability
            ],
            evidence=['load', 'real_failure'],
            evidence_card=[2, 2]
        )
        self.model.add_cpds(cpd_carico, cpd_guasto, cpd_ml)

    def get_real_failure_probability(self, ml_alert, sensor_load_value):
        """
        Calcola la probabilità REALE di guasto combinando ML e sensori di contesto.
        sensor_load_value: 1 se il sensore os1/os2 indica sforzo elevato, 0 altrimenti.
        """
        # Eseguiamo l'inferenza probabilistica
        evidenza = {
            'ML_Output': 1 if ml_alert else 0,
            'load': 1 if sensor_load_value > 0.8 else 0 # Soglia di carico
        }
        
        result = self.inference.query(variables=['real_failure'], evidence=evidenza)
        prob_guasto = result.values[1]
        
        return prob_guasto


"""
    def plot_network(self):
        #Visualizza il grafo della rete bayesiana.

        nx_graph = nx.DiGraph(self.model.edges())
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=12)
        plt.title("Bayesian Network Structure")
        plt.show()

if __name__ == "__main__":
    # Dummy KG module con metodo get_task_context che restituisce un dizionario minimo
    class DummyKG:
        def get_task_context(self, component):
            return {"unit": "unit_1"}  # o "unit_2" per vedere la differenza

    kg = DummyKG()
    bn = UncertaintyModule(kg, "hpc")
    bn.plot_network()
"""


