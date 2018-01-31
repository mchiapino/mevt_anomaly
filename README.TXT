## Détection des sous-groupes de variables dépendantes dans les extrêmes
et estimation du mélange de Dirichlets.
-> flight_data_EM_script.py

-Les sous-groupes de features sont détectés par DAMEX ou par CLEF.
La variable R représente le seuil au dessus duquel on estime qu'un point est 'extreme'.
kappa_min est le critère d'arrêt qui détermine les groupes de features que l'on garde
(plus kappa_min est grand, moins l'on gardera de sous-groupes).
-Les sous-groupes de features determinent le support du mélange de Dirichlets (pour
chaque sous-groupe une distribution de Dirichlet y est associée).
L'algorithme EM estime les paramètres du mélange.
La variable qui nous intéresse au final est 'gamma_z', gamma_z[i, k] est la probabilité
que le point i appartiennent au sous-groupe k.


## Clustering spectral et visualisation du graphe.
-> visualisation.py

-A partir de gamma_z nous pouvont construire la matrice d'adjacence W pour faire du
clustering sur les points (les vols 'extreme'), avec un algorithme de clustering spectral.
La variable K_spec détermine le nombre de cluster final au sortir de l'algorithme.
Les clusters de vols sont dans la variable 'flights_cluster' (liste de liste d'index de vols
(index des données de départ)) et les clusters de features utilisé pour construire le
mélange de Dirichlets sont dans 'flights_parameters_clusters'.

