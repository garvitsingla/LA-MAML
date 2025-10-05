# plot_env.py
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

# global font sizes
plt.rcParams.update({
    "font.size": 14,        
    "axes.titlesize": 16,   
    "axes.labelsize": 14,  
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

def load_env_data(env_dir):
    maml_path = os.path.join(env_dir, "maml_avg_steps.npy")
    lang_path  = os.path.join(env_dir, "lang_avg_steps.npy")
    if not (os.path.exists(maml_path) and os.path.exists(lang_path)):
        raise FileNotFoundError(f"Missing npy files in {env_dir}")
    maml = np.load(maml_path)
    lang  = np.load(lang_path)
    env_name = os.path.basename(os.path.normpath(env_dir))
    return maml, lang, env_name
def plot_line(env_dir, save_path=None, show=True, dpi=300, out_dir="figures"):
    maml, lang, env_name = load_env_data(env_dir)

    plt.plot(maml, label="MAML Policy")
    plt.plot(lang,  label="Lang-adapted Policy")
    plt.xlabel("Meta-iterations")
    plt.ylabel("Average Steps")
    plt.title(env_name)
    plt.legend()
    plt.grid(alpha=0.4)
    plt.tight_layout()

    # save path
    if save_path is None:
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f"{env_name}.png")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    print(f"Saved {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MAML vs Lang steps for one environment")
    parser.add_argument("env_dir", help="Path to environment folder (e.g., metrics/GoToLocal)")
    parser.add_argument("--save", metavar="FILE", help="Save figure to file (PNG, PDF, etc.)")
    parser.add_argument("--dpi", type=int, default=400, help="DPI for saved figure")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    args = parser.parse_args()

    plot_line(args.env_dir, save_path=args.save, show=not args.no_show, dpi=args.dpi)