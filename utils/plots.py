import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


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
    
    

def plot_roc_curves(trained_models, test_dataset):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 18)) 
    fig.suptitle('Receiver Operating Characteristic (ROC) Curves for Tested Models')
    
    keys = list(trained_models.keys())
    
    for i, ax in enumerate(axes.flat):
        if i < len(keys):
            models_dict = trained_models[keys[i]]
            for model_name, model in models_dict.items():
                if hasattr(model, "predict_proba"): 
                    y_score = model.predict_proba(test_dataset.X)[:, 1]
                    fpr, tpr, _ = roc_curve(test_dataset.y, y_score)
                    roc_auc = auc(fpr, tpr)
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
                else:
                    print(f"Model {model_name} does not support predict_proba and will be skipped.")
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'Models for {keys[i]}')
            ax.legend(loc="lower right")
        else:
            fig.delaxes(ax) 

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

