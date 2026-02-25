# Vision3D_test

Mini app Django de reconstruction/panorama avec visualisation 3D.

## Lancer en dev

```bash
source .venv/bin/activate
python manage.py migrate
python manage.py runserver
```

Puis ouvrir http://127.0.0.1:8000/

## Nettoyer la base locale

```bash
rm -f db.sqlite3
python manage.py migrate
```

## Nettoyer les reconstructions déjà faites

```bash
rm -rf media/jobs/* media/staging/*
```

Optionnel (si tu veux tout remettre propre d'un coup) :

```bash
rm -f db.sqlite3
rm -rf media/jobs/* media/staging/*
python manage.py migrate
```
