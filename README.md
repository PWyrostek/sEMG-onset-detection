# sEMG-onset-detection

## Wymagania
-python3\
-scipy\
-scipy.io\
-numpy\
-matplotlib.pyplot\
-math\
-multiprocessing\
-sklearn.decomposition

## Instrukcja uruchomienia programu
Do folderu z plikiem modeling.py konieczne jest pobranie z repozytorium pliku database.mat.\
W funkcji main() do wartości emg_data przypisujemy odwołujemy się do jakiejś z wartości 'emg1' do 'emg20', która odpowiada odpowiedniej tabeli z bazy database.mat\
do wartości data_column przypisujemy wartość od 0 do 5 odpowiadającą interesującej nas kolumnie z tabeli z danymi emg\

## Funkcje odpowiadające za wykrywanie onset
-onset_two_step_alg(data, W_1, k_1, d_1, h_2, W_2, M_2) gdzie data to pojedyncza kolumna z tabeli z danymi emg, argumenty W_1, k_1 oraz d_1 to argumenty algorytmu onset_sign_changes, a argumenty h_2, W_2, M_2 to parametry algorytmu onset_AGLRstep\

-onset_sign_changes(data, W, k, d) gdzie data to pojedyncza kolumna z tabeli z danymi emg, argument W to wielkość okna, argumenty k i d to odpowiednio mnożnik i stała na podstawie której obliczana jest wartość h odpowiadająca za czułość całego algorytmu\

-onset_AGLRstep(data, h, W, M) gdzie data to pojedyncza kolumna z tabeli z danymi emg, argument W to wielkość okna, argument h to wartość odpowiadająca za czułość algorytmu (odczyty mniejsze niż h są pomijane) oraz argument M odpowiadający za ilość pierwszych odczytów sygnału, z których obliczana jest wartość theta_0\

-onset_hodges_bui(data, h, W, M) gdzie data to pojedyncza kolumna z tabeli z danymi emg, argument W to wielkość okna, argument h to wartość odpowiadająca za czułość algorytmu (odczyty mniejsze niż h są pomijane) oraz argument M odpowiadający za ilość pierwszych odczytów sygnału, z których obliczana jest średnia oraz odchylenie standardowe\

-onset_komi(data, h) gdzie data to pojedyncza kolumna z tabeli z danymi emg, a argument h to wartość odpowiadająca za czułość algorytmu (odczyty mniejsze niż h są pomijane)

## Funkcja szukająca najlepszych parametrów
Do funkcji find_optimal_params przekazujemy całą bazę danych (zmienną mat_data) oraz nazwę funkcji testowej dla algorytmu, dla którego chcemy znaleźć parametry (np. function_test_twostep, function_test_komi itp.)

## Tworzenie wykresu
Za tworzenie wykresu odpowiada funkcja make_plot(emg_data, torque_data, expected, found_onset=0, found_right_side=5000)\
-emg_data - dane z kolumny tabeli emg\
-torque_data - dane z kolumny nr 7 z tabeli emg\
-expected - ustalona przez specjalistów wartość onset dla przekazanych danych\
-found_onset - znaleziona przez nas wartość onset\
Na wykresie zielonym kolorem oznaczona jest linia z oczekiwaną wartością, a pomarańczowym kolorem linia ze znalezioną wartością
