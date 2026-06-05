import SearchingSolutions 

class CSP: 

    def __init__(self):
        #Domain
        self.technicians = [
            {'id':'Tech1','skills': ['Area_A', 'Area_B', 'Test_Area'], 'level': 'Senior'},
            {'id':'Tech2','skills': ['Area_A','Area_B'], 'level': 'Junior'},
            {'id':'Tech3','skills': ['Test_Area'], 'level': 'Specialist'},
            {'id':'Tech4','skills': ['Area_A','Area_B','Test_Area'], 'level': 'Senior'}
        ]

        #Variables
        self.Tasks = [
            {'id': 'T1', 'area': 'Area_A', 'priority': 'High'},
            {'id': 'T2', 'area': 'Area_B', 'priority': 'Medium'},
            {'id': 'T3', 'area': 'Test_Area', 'priority': 'High'},
            {'id': 'T4', 'area': 'Area_A', 'priority': 'Low'},
            {'id': 'T5', 'area': 'Area_B', 'priority': 'Low'}
        ]

    def is_valid(self, technician_id, task_id, current_assignments):
        # Controlla se il tecnico ha le competenze per il task
        technician = next(t for t in self.technicians if t['id'] == technician_id)
        task = next(t for t in self.Tasks if t['id'] == task_id)
        
        # Vincolo di competenze
        if task['area'] not in technician['skills']:
            return False
        
        # Vincolo di carico : Verifica carico di lavoro , se il tecnico ha già 1 task assegnato, non può prenderne altri
        # Controlliamo se l'ID del tecnico è già presente tra i valori del dizionario
        if technician_id in current_assignments.values():
            return False
        
        # Vincolo di Seniority (Ragionamento Logico)
        # Se il task richiede un livello specifico 
        if 'required_level' in task:
            req = task['required_level']
        
            # Logica gerarchica: Un Senior può fare tutto. 
            # Un Junior può fare solo task Junior.
            if req == 'Senior' and technician['level'] != 'Senior':
                return False
            
            if req == 'Specialist' and technician['level'] != 'Specialist':
                return False
        
        return True
    

    # Backtracking Search Algorithm 
    def backtracking_search(self, tasks, index, current_assignments):
        # CASO BASE: Se l'indice è uguale alla lunghezza delle task, abbiamo finito!
        if index == len(tasks):
            return current_assignments
        
        # Selezioniamo la task corrente in base all'indice
        task = tasks[index]
        
        for technician in self.technicians:
            tech_id = technician['id']
            
            if self.is_valid(tech_id, task['id'], current_assignments):
                # PROVA: Assegna il task al tecnico
                current_assignments[task['id']] = tech_id
                
                # RICORSIONE: Passa alla task successiva (index + 1)
                result = self.backtracking_search(tasks, index + 1, current_assignments)
                
                if result is not None:
                    return result
                
                # BACKTRACK: Se il ramo non porta a una soluzione, annulla l'assegnazione
                del current_assignments[task['id']]
        
        return None

    def solve(self):
        # 1. ORDINAMENTO: Prima di tutto, mettiamo in fila le urgenze
        priority_order = {'High': 3, 'Medium': 2, 'Low': 1}
        # Creiamo una versione ordinata delle Tasks per non alterare l'originale
        sorted_tasks = sorted(self.Tasks, key=lambda x: priority_order[x['priority']], reverse=True)
        
        # 2. AVVIO: Lanciamo la ricerca partendo dall'indice 0
        return self.backtracking_search(sorted_tasks, 0, {})
