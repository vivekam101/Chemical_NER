import requests
import pickle

with open("chemical_names.pkl", "rb") as f:
    chemical_names = pickle.load(f)
for topic in chemical_names:
    response = requests.get(
                'https://en.wikipedia.org/w/api.php',
                params={
                    'action': 'query',
                    'format': 'json',
                    'titles': topic ,
                    'prop': 'extracts',
                    'exintro': True,
                    'explaintext': True,
                    }
                ).json()
    page = next(iter(response['query']['pages'].values()))
    try:
        print(page['extract'])
    except KeyError:
        continue
