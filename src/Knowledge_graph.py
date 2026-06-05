from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS

class FactoryKG:
    def __init__(self):
        self.g = Graph()
        self.EX = Namespace("http://example.org/plant/")
        self.g.bind("ex", self.EX)
        self._build_ontology()

    def _build_ontology(self):
        EX = self.EX
        
        # 1. GERARCHIA DELLE CLASSI (Tutte le relazioni) 
        # Definiamo che tutti i tipi di tecnico sono sottoclassi di Technician
        self.g.add((EX.Senior, RDFS.subClassOf, EX.Technician))
        self.g.add((EX.Junior, RDFS.subClassOf, EX.Technician))
        self.g.add((EX.Specialist, RDFS.subClassOf, EX.Technician))
        
        # Definiamo le macchine e i componenti
        self.g.add((EX.Engine, RDFS.subClassOf, EX.Machine))
        self.g.add((EX.Component, RDFS.subClassOf, EX.Part))

        # 2. REPARTI 
        depts = [EX.Area_A, EX.Area_B, EX.Warehouse, EX.Test_Area]
        for d in depts:
            self.g.add((d, RDF.type, EX.Department))

        # 3. MOTORI E COLLOCAZIONE
        # Unit 1 (Area A) e Unit 2 (Area B)
        self.g.add((EX.unit_1, RDF.type, EX.Engine))
        self.g.add((EX.unit_1, EX.located_in, EX.Area_A))
        self.g.add((EX.unit_2, RDF.type, EX.Engine))
        self.g.add((EX.unit_2, EX.located_in, EX.Area_B))

        # 4. COMPONENTI (Tutte le Unit)
        # Unit 1
        comp1 = ["hpc", "hpt", "fan", "injector", "pressure_sensor"]
        for c in comp1:
            self.g.add((EX[c], RDF.type, EX.Component))
            self.g.add((EX[c], EX.part_of, EX.unit_1))
            self.g.add((EX[c], EX.is_physical_part, Literal(True)))

        # Unit 2
        comp2 = ["lpc", "lpt", "oil_pump", "shaft"]
        for c in comp2:
            self.g.add((EX[c], RDF.type, EX.Component))
            self.g.add((EX[c], EX.part_of, EX.unit_2))
            self.g.add((EX[c], EX.is_physical_part, Literal(True)))

        # 5. TECNICI E LIVELLI
        # Assegniamo ogni tecnico al suo livello e alle sue aree (Skills)
        # Tech 1: Senior
        self.g.add((EX.Tech1, RDF.type, EX.Senior))
        for area in [EX.Area_A, EX.Area_B, EX.Test_Area]:
            self.g.add((EX.Tech1, EX.has_skill, area))

        # Tech 2: Junior
        self.g.add((EX.Tech2, RDF.type, EX.Junior))
        for area in [EX.Area_A, EX.Area_B]:
            self.g.add((EX.Tech2, EX.has_skill, area))

        # Tech 3: Specialist
        self.g.add((EX.Tech3, RDF.type, EX.Specialist))
        self.g.add((EX.Tech3, EX.has_skill, EX.Test_Area))

    def get_task_context(self, component_name):

        # Recupera le informazioni dal grafo per pianificare il movimento.
        comp_uri = self.EX[component_name]
        query = """
        PREFIX ex: <http://example.org/plant/>
        SELECT ?unit ?area ?is_physical WHERE {
            ?comp ex:part_of ?unit .
            ?unit ex:located_in ?area .
            OPTIONAL { ?comp ex:is_physical_part ?is_physical } .
        }
        """
        results = self.g.query(query, initBindings={'comp': comp_uri})
        for row in results:
            return {
                "unit": str(row.unit).split('/')[-1],
                "area": str(row.area).split('/')[-1],
                "needs_warehouse": bool(row.is_physical)
            }
        return None