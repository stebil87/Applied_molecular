import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def plot_tsne(dfs, titles):
    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    axes = axes.ravel()

    for i, df in enumerate(dfs):
        df_copy = df.copy()
        X = df_copy.drop(['BBB', 'smiles'], axis=1)  
        y = df_copy['BBB']
        
        tsne = TSNE(n_components=2, random_state=42)
        X_embedded = tsne.fit_transform(X)

        ax = axes[i]
        ax.scatter(X_embedded[y == 0, 0], X_embedded[y == 0, 1], c='blue', label='Class 0', alpha=0.5)
        ax.scatter(X_embedded[y == 1, 0], X_embedded[y == 1, 1], c='red', label='Class 1', alpha=0.5)
        ax.set_title(titles[i])
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.legend()

    plt.show()
