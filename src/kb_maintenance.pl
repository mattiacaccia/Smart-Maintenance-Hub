% --- GERARCHIA COMPONENTI ---

% Unit 1 (Area A)
part_of(hpc, engine_1).
part_of(hpt, engine_1).
part_of(fan, engine_1).
part_of(injector, engine_1).
part_of(pressure_sensor, engine_1).

% Unit 2 (Area B)
part_of(lpc, engine_2). % Low Pressure Compressor
part_of(lpt, engine_2). % Low Pressure Turbine
part_of(oil_pump, engine_2).
part_of(shaft, engine_2).

% Warehouse
% Definiamo quali componenti sono fisici e richiedono un ricambio
needs_spare_part(hpc).
needs_spare_part(hpt).
needs_spare_part(lpc).
needs_spare_part(lpt).
needs_spare_part(shaft).
needs_spare_part(fan).
needs_spare_part(oil_pump).

% --- CARATTERISTICHE ---

% tecnical complexity: definisce quali componenti sono complessi e richiedono un tecnico senior
complex(hpc).
complex(hpt).
complex(lpt).
complex(shaft).

% --- REGOLE DI SENIORITY ---

% Richiede Senior se il pezzo è complesso
requires_senior(Part) :- complex(Part).

% Richiede Senior se lurgenza è estrema (< 5 cicli/ore)
requires_senior_urgency(E) :- 
    failure_prediction(E, T), 
    T < 5.

% Regola finale per il livello tecnico
required_level(E, Part, 'Senior') :- 
    requires_senior(Part) ; requires_senior_urgency(E).
required_level(_, _, 'Junior'). % Default se non serve un senior

% --- LOGICA TEST AREA ---

% Un test è obbligatorio se è stato toccato un pezzo complesso
requires_special_test(E) :- 
    part_of(Part, E),
    complex(Part).

% Regola per l Area di Test
assign_test_specialist(E) :- requires_special_test(E).

% Un task va in test area se richiede un test speciale (pezzo complesso)
goes_in_test_area(E, Part) :-
    part_of(Part, E),
    complex(Part).

% --- LOGICA WAREHOUSE ---

% Regola: Il tecnico deve passare in magazzino se il pezzo è un ricambio fisico
requires_warehouse_stop(Part) :- needs_spare_part(Part).

% Una manutenzione è urgente se la predizione di guasto è inferiore a una certa soglia (es: 24 cicli)
urgent_maintenance(E) :-
    failure_prediction(E, T),
    T < 24.

% --- LOGICA DECISIONALE COMPLESSA ---
% Determina se serve un tecnico Senior basandosi su complessità O urgenza predetta dal ML
check_senior_needed(_, Part) :- 
    complex(Part).
check_senior_needed(Unit, _) :- 
    failure_prediction(Unit, RUL), 
    RUL < 15. % Se il ML predice meno di 15 cicli, il caso è critico -> serve un Senior

% Regola per decidere se è necessario il passaggio in Test Area
% Non solo per i pezzi complessi, ma anche se la RUL è molto bassa (controllo sicurezza)
needs_test_logic(_, Part) :- 
    complex(Part).
needs_test_logic(Unit, _) :- 
    failure_prediction(Unit, RUL), 
    RUL < 10.

% --- REGOLA DI VALIDAZIONE COMPONENTI---
% Verifica se una parte appartiene correttamente a un unità
is_compatible(Unit, Part) :-
    part_of(Part, Unit).

% Se vogliamo una regola che ci dia l errore esplicito
check_mapping_error(Unit, Part) :-
    \+ is_compatible(Unit, Part).

