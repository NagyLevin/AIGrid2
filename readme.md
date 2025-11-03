-1 → WALL (fal, áthatolhatatlan).

0 → EMPTY / ROAD (járható pálya).

1 → START (rajtcella, járható).

100 → GOAL (célcella/finish, járható) – ez a te kivágatodban épp nem szerepel, de létezik.

2 → UNKNOWN (belső jelölés ismeretlenre; a judge-kivágatokban általában nem találkozol vele).

3 → NOT_VISIBLE (köd/fog-of-war: a látótávon kívüli cellák a kapott 2R+1 × 2R+1 ablakban).

PL
Agent: 0 1
JUDGE: 3 3 0 1
JUDGE: 2 4
JUDGE: 3 3

Agent: 0 1 → az gyorsulási vektor (ax=0, ay=+1). Nem közvetlen pozíció, hanem a sebesség változása.
JUDGE: 3 3 0 1 ⇒ az új állapotod: pozíció = (3,3), sebesség = (0,1).
JUDGE: 2 4 → az egyik játékos pozíciója: (2,4).
JUDGE: 3 3 → egy másik játékos pozíciója: (3,3).

Elvileg ezek globális poziciók


idea 1
https://stackoverflow.com/questions/34001535/path-finding-with-limited-vision
