import cPickle
import os
import traceback
from itertools import islice

import numpy as np
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from scipy.optimize import nnls, differential_evolution
from scipy.sparse import load_npz
from scipy.sparse.linalg import norm as sparse_norm

NULL_SUB = ''
# Number of subs to display on the algebra page
N_SUBS = 3
# Number of results to return on the algebra page on page-load, refinement, and search auto-completion
N_RESULTS = 20

DATA_DIR = os.path.join(settings.BASE_DIR, 'data')


# Only want to load the data once on the server
# Couldn't find a good way to do this at server start-up, so stuck with this for now
def run_once(f):
    def wrapper():
        if not wrapper.has_run:
            wrapper.result = f()
            wrapper.has_run = True
        return wrapper.result

    wrapper.has_run = False
    return wrapper


# Loads the matrix of subreddit vectors
# X is a sparse, non-negative matrix with 40,875 rows and 1,800 columns
# X[i] is the ith row of this matrix and it represents the ith subreddit
@run_once
def get_X():
    return load_npz(os.path.join(DATA_DIR, 'X.npz'))


# Sorted list of sub names by their popularity in descending order (40,875 entries)
@run_once
def get_subs_by_popularity():
    with open(os.path.join(DATA_DIR, 'subs_by_popularity.pkl'), 'rb') as f:
        obj = cPickle.load(f)
    return obj


# Convert a subreddit by name to its corresponding row index in X
@run_once
def get_sub_to_index():
    with open(os.path.join(DATA_DIR, 'sub_to_index.pkl'), 'rb') as f:
        obj = cPickle.load(f)
    return obj


# Inverse of sub_to_index
@run_once
def get_index_to_sub():
    with open(os.path.join(DATA_DIR, 'index_to_sub.pkl'), 'rb') as f:
        obj = cPickle.load(f)
    return obj


# The score of a solution to the map problem
# z = Ax is our guess
# The score is 1 / |z| * dot(z, x)
def score(A, x, y):
    norm = np.linalg.norm(A.dot(x))
    if norm == 0:
        return 0
    val = A.dot(x).T.dot(y) / norm
    return val


# Solve the map problem using non-negative least squares
def solve_nnls(A, y):
    z = nnls(A, y)[0]
    return z


# Solve the map problem using 538's method
def solve_538(A, y):
    z = A.T.dot(y)
    return z


def solve_global(A, y):
    fn = lambda x: -score(A, x, y)
    results = differential_evolution(fn, [(0.0, 1.0)] * A.shape[1])
    print('DE results', results)
    z = results.x
    return z


def algebra_view(request):
    subs = [request.GET.get('sub{}'.format(subnum), NULL_SUB) for subnum in range(N_SUBS)]
    subsset = set(subs)
    ops = [request.GET.get('op{}'.format(subnum), 'plus') for subnum in range(N_SUBS - 1)]
    if all(sub == NULL_SUB for sub in subs):
        return render(request, 'algebra.html', {
            'subs': subs, 'ops': ops, 'view': 'algebra'
        })

    # Combine each subreddit's vector using the defined operations
    X = get_X()
    sub_to_index = get_sub_to_index()
    if subs[0] != NULL_SUB:
        vec = X[sub_to_index[subs[0]]]
    else:
        vec = np.zeros_like(X[0])
    for sub, op in zip(subs[1:], ops):
        if sub != NULL_SUB:
            if op == 'plus':
                vec = vec + X[sub_to_index[sub]]
            elif op == 'minus':
                vec = vec - X[sub_to_index[sub]]
            else:
                return render(request, 'algebra.html', {
                    'subs': subs, 'ops': ops, 'error': 'Invalid operation ({})'.format(op), 'view': 'algebra'
                })

    # Need to renormalize this vector because a normalized vector plus/minus a normalized vector is not normalized
    norm = sparse_norm(vec)
    if norm != 0:
        vec = vec / norm

    # Get the similarity of this vector to every subreddit vector
    sims = X.dot(vec.T).toarray().ravel().astype(np.float16)
    # Sort the results in descending order
    sorted_indices = sims.argsort().astype(np.uint16)[::-1]
    index_to_sub = get_index_to_sub()
    # Store the results in the session so that they can be used in the refined search
    request.session['sorted_indices'] = sorted_indices
    request.session['sims'] = sims
    # Limit the results and don't include subreddits used in the initial algebra
    subsims = ((index_to_sub[i], round(sims[i] * 100, 1)) for i in sorted_indices if index_to_sub[i] not in subsset)
    subsims = list(islice(subsims, N_RESULTS))
    return render(request, 'algebra.html', {
        'subs': subs, 'subsims': subsims, 'ops': ops, 'view': 'algebra'
    })


# Renders the map view
# This page shows where a bunch of subreddits lie in between other subreddits
def map_view(request):
    insubs = request.GET.getlist('insubs', [])[:10]
    outsubs = request.GET.getlist('outsubs', [])[:10]
    method = request.GET.get('method', 'nnls')

    # Doesn't make sense to draw a graph with less than two outer subreddits
    # It would either be a point, or some undefined thing
    if len(outsubs) < 2:
        return render(request, 'map.html', {
            'insubs': insubs,
            'outsubs': outsubs,
            'view': 'map',
            'method': method
        })

    X = get_X()
    sub_to_index = get_sub_to_index()
    invecs = X[np.array([sub_to_index[sub] for sub in insubs])].toarray()
    outvecs = X[np.array([sub_to_index[sub] for sub in outsubs])].toarray()

    A = outvecs.T

    # Generate the outer vertices of the regular n-gon
    n_outvecs = len(outsubs)
    dtheta = np.pi * 2 / n_outvecs
    outpoints = []
    for i, sub in enumerate(outsubs):
        theta = dtheta * i
        x = np.cos(theta)
        y = np.sin(theta)
        outpoints.append((x, y, sub))
    outpoints_str = ' '.join('{},{}'.format(x1, x2) for x1, x2, sub in outpoints)

    # Solve the map problem for each inner subreddit
    inpoints = []
    legend = []
    dh = 1.0 / len(invecs)
    for i, (sub, y) in enumerate(zip(insubs, invecs)):
        fill = 'hsl(' + str(dh * i * 360) + ', 100%, 50%)'
        if method == '538':
            z = solve_538(A, y)
        else:
            method = 'nnls'
            z = solve_nnls(A, y)
        z = z / np.sum(z)
        sim = score(A, z, y)
        sim = str(round(sim * 100, 1)) + '%'
        x = sum(x2 * v for (x2, y2, _), v in zip(outpoints, z))
        y = sum(y2 * v for (x2, y2, _), v in zip(outpoints, z))
        inpoints.append((x, y, fill))
        legend.append((dh * (i + 0.5), sub, sim, fill))

    return render(request, 'map.html', {
        'insubs': insubs,
        'outsubs': outsubs,
        'inpoints': inpoints,
        'legend': legend,
        'outpoints': outpoints,
        'outpoints_str': outpoints_str,
        'view': 'map',
        'method': method
    })


# Display the about page
def about_view(request):
    return render(request, 'about.html', {'view': 'about'})


# AJAX view used in the algebra page for refining results by a search term
def refine_view(request):
    q = request.GET.get('query', '')
    subsset = set(request.GET.get('sub{}'.format(subnum), NULL_SUB) for subnum in range(N_SUBS))
    sorted_indices = request.session.get('sorted_indices', None)
    sims = request.session.get('sims', None)
    if sorted_indices is None or sims is None:
        # This can happen if MemCachier (our in-memory cache service) runs out of space
        # Given that we're storing about 40,875 records * (8 B + 8 B) / record
        # We can only store about 100 result sets at a time because we have 25MB on the free tier
        # MemCachier seems to flush itself in an approximate LRU order (seems like a clock cycle)
        # This shouldn't be a rare occurence, but we still need to tell the client that we can't get their results
        return HttpResponse('')
    index_to_sub = get_index_to_sub()
    subsims = ((index_to_sub[i], round(sims[i] * 100, 1)) for i in sorted_indices
               if q.lower() in index_to_sub[i].lower() and index_to_sub[i] not in subsset)
    subsims = list(islice(subsims, N_RESULTS))
    response = HttpResponse(render_to_string('table.html', {'subsims': subsims}))
    return response


# AJAX view used in both the algebra page and the map page for fetching autocomplete results
def search_view(request):
    q = request.GET.get('query', '').lower()
    items = []
    for sub, n_authors in get_subs_by_popularity():
        if len(items) > N_RESULTS:
            break
        if q.lower() in sub.lower():
            items.append({'text': sub, 'id': sub})
    return JsonResponse({'items': items})


# Enable logging whenever a server error occurs
def server_error(request):
    print('Server Error. Printing stack trace.')
    traceback.print_exc()
    return render(request, '500.html')
