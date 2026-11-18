"""Explicit reference epochs; stock serial is not silently replaced by a patched fork."""
import copy
from compression_pair_cases import suite_spec

REFERENCES=('ordered-copy','stock-serial')


def reference_config(suite,reference):
    if reference not in REFERENCES:raise ValueError(f'Unknown ndzip reference {reference}')
    config=copy.deepcopy(suite_spec(suite))
    if reference=='stock-serial':
        routes=[]
        for route in config['routes']:
            if route['codec']=='ndzip':route['cpu_threads']=1
            if route not in routes:routes.append(route)
        config['routes']=routes
    return config
