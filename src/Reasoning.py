from pyswip import Prolog

class ReasoningModule:
    def __init__(self, pl_file="kb_maintenance.pl"):
        """
        Inizializza il motore Prolog e carica il file delle regole.
        """
        self.prolog = Prolog()
        try:
            self.prolog.consult(pl_file)
            print(f"LOGIC SYSTEM: File '{pl_file}' loaded successfully.")
        except Exception as e:
            print(f"ERROR: Unable to load Prolog file. Check the path. \n{e}")

    def get_requirements(self, unit_id, part_id, current_rul):
    # 1. Normalizzazione degli ID per Prolog
        # Se l'utente seleziona 'Unit_1', nel Prolog cerchiamo 'engine_1'
        engine_id_prolog = unit_id.lower().replace("unit_", "engine_")
        
        u_id = f"'{unit_id}'"       # Per la predizione RUL
        e_id = f"'{engine_id_prolog}'" # Per il controllo di appartenenza
        p_id = f"'{part_id}'"
        
        # 2. CONTROLLO DI COMPATIBILITÀ
        # Chiediamo al Prolog: "La parte X appartiene al motore Y?"
        # query: part_of(oil_pump, engine_2)
        is_valid = bool(list(self.prolog.query(f"part_of({p_id}, {e_id})")))
        
        if not is_valid:
            # Se metti Unit_1 e oil_pump, is_valid sarà False -> ERRORE
            return {"error": f"Incoerenza rilevata: Il componente '{part_id}' non appartiene alla '{unit_id}'."}

        # FASE 1: AGGIORNAMENTO DATI
        # Cancelliamo la vecchia predizione per quell'unità(motore)
        self.prolog.retractall(f"failure_prediction({u_id}, _)")
        # Inseriamo la nuova predizione RUL (dato che arriva dal modello ML)
        self.prolog.assertz(f"failure_prediction({u_id}, {current_rul})")

        # FASE 2: INTERROGAZIONE (QUERY)
        # Chiediamo: serve un Senior? 
        is_senior = bool(list(self.prolog.query(f"check_senior_needed({u_id}, {p_id})")))

        # Chiediamo: la manutenzione è urgente?
        is_urgent = bool(list(self.prolog.query(f"urgent_maintenance({u_id})")))

        # Chiediamo: dopo la riparazione deve andare in Area di Test? 
        needs_test = bool(list(self.prolog.query(f"needs_test_logic({u_id}, {p_id})")))

        # Chiediamo : serve passare in magazzino per i pezzi di ricambio?
        needs_warehouse = bool(list(self.prolog.query(f"requires_warehouse_stop({p_id})")))

        # FASE 3: RISULTATO (Sincronizzato con Main e CSP)
        return {
            'is_senior': is_senior,           # Usato dal CSP per il vincolo 'required_level'
            'priority': 'High' if is_urgent else 'Medium', # Usato dal CSP per l'ordinamento task
            'needs_test': needs_test,         # Usato da A* per pianificare la sosta in Test_Area
            'needs_warehouse': needs_warehouse, # Usato da A* per la sosta in Warehouse
            'target_engine': unit_id,
            'component': part_id,
            'current_rul': current_rul
        }