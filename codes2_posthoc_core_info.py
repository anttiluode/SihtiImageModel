"""POST-HOC (not pre-registered): how much class information does the 16-number core hold at all?
If a nonlinear reader of the core alone beats the fills' judge accuracy, the wrong readings are not an
information limit; the information is there and the fill fails to use it."""
import os, pickle, warnings
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sihti_model.data import load
warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
for ds in ["mnist", "fashion"]:
    m = pickle.load(open(f"results/cache/model_{ds}.pkl", "rb"))
    Xtr, ytr = load(ds, "train"); Xte, yte = load(ds, "test")
    ztr = np.concatenate([m.encode(Xtr[i:i + 10000])[4] for i in range(0, 60000, 10000)]); zte = m.encode(Xte)[4]
    mu, sd = ztr.mean(0), ztr.std(0); ztr, zte = (ztr - mu) / sd, (zte - mu) / sd
    mlp = MLPClassifier((256, 256), max_iter=60, random_state=0).fit(ztr, ytr)
    knn = KNeighborsClassifier(15).fit(ztr, ytr)
    print(ds, "core only: backprop MLP", round(mlp.score(zte, yte), 4), " kNN (closed form)", round(knn.score(zte, yte), 4))
