import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
plt.rc("text",usetex=True)

# Define constants and matrix
R = 4
x_max=6
fs = 16
divide = 100
G = np.array([[1.5, 0.2], [0.1, 0.5]])
G1 = G[0, 0]**2 + G[0, 1]**2
G2 = G[1, 0]**2 + G[1, 1]**2
G3 = G[0, 0] * G[1, 0] + G[0, 1] * G[1, 1]

# Calculate terms for C
C1 = 3 * G[0, 0]**4 + 6 * G[0, 0]**2 * G[0, 1]**2 + 3 * G[0, 1]**4
C2 = 3 * G[1, 0]**4 + 6 * G[1, 0]**2 * G[1, 1]**2 + 3 * G[1, 1]**4
C3 = 2 * R**2 * G1
C4 = 2 * R**2 * G2
C5 = 3 * (G[0, 0]**2 * G[1, 0]**2 + G[0, 1]**2 * G[1, 1]**2) + 4 * G[0, 0] * G[0, 1] * G[1, 0] * G[1, 1] + (G[0, 1]**2 * G[1, 0]**2 + G[0, 0]**2 * G[1, 1]**2)

# Final C value
C = R**4 + C1 + C2 + C3 + C4 + C5 * 2

# Define r_1(x) function
def r_1(x1, x2):
    return (x1**2 + x2**2 - R**2)**2

# Define \tilde{r}_1(y) function
def r_tilde_1(y1, y2):
    term1 = y1**4 - y1**2 * (6 * G1 + 2 * G2 + 2 * R**2)
    term2 = y2**4 - y2**2 * (6 * G2 + 2 * G1 + 2 * R**2)
    term3 = 2 * y1**2 * y2**2 - 8 * y1 * y2 * G3
    return term1 + term2 + term3 + C

def r_1m(x1, x2):
    return x1**4+2*x1**2*x2**2+x2**4+\
           2*(x1**2*(3*G1+G2)+x2**2*(G1+3*G2))+\
           2*(G[0, 0]**2 * G[1, 0]**2 + G[0, 1]**2 * G[1, 1]**2)+\
           2*(G[0, 1]**2 * G[1, 0]**2 + G[0, 0]**2 * G[1, 1]**2)+\
           6*(G[0, 0]**4 + G[0, 1]**4 + G[1, 0]**4 + G[1, 1]**4)+\
           2*(G[0, 0]**2 * G[0, 1]**2+G[0, 1]**2 * G[1, 1]**2)-\
           2*R**2*(x1**2+x2**2+(G1+G2))+\
           R**4

def figure5():
    # Create grid data
    x1_vals = np.linspace(-x_max, x_max, 400)
    x2_vals = np.linspace(-x_max, x_max, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Compute values of r_1 and \tilde{r}_1
    Z_r1 = r_1(X1, X2)
    Z_r_tilde_1 = r_tilde_1(X1, X2)

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Adjust linthresh for more detail around zero
    linthresh = 1  # Increase linthresh to have more detail around zero
    

    # Normalize color scale using SymLogNorm for each plot
    norm_r1 = colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=Z_r1.min(), vmax=Z_r1.max(), base=10)
    norm_r_tilde_1 = colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=Z_r_tilde_1.min(), vmax=Z_r_tilde_1.max(), base=10)



    # Plot r_1(x)
    c1 = axs[0].contourf(X1, X2, Z_r1, levels=divide, cmap='coolwarm', norm=norm_r1)
    axs[0].set_title('$r_1(x)$', fontsize=fs)
    axs[0].set_xlabel('$x_1$', fontsize=fs)
    axs[0].set_ylabel('$x_2$', fontsize=fs)
    axs[0].set_xlim([-x_max, x_max])
    axs[0].set_ylim([-x_max, x_max])
    cbar1 = fig.colorbar(c1, ax=axs[0], orientation='vertical')
    cbar1.set_label('Value (SymLog scale)', fontsize=fs)

    # Use scientific notation for the colorbar ticks
    cbar1.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()
    # Plot the circle on the first subplot
    circle = plt.Circle((0, 0), R, color='black', linestyle='--', fill=False, lw=2)
    axs[0].add_patch(circle)

    # Plot \tilde{r}_1(y)
    c2 = axs[1].contourf(X1, X2, Z_r_tilde_1, levels=divide, cmap='coolwarm', norm=norm_r_tilde_1)
    axs[1].set_title('$\\tilde{r}_1(y)$', fontsize=fs)
    axs[1].set_xlabel('$y_1$', fontsize=fs)
    axs[1].set_ylabel('$y_2$', fontsize=fs)
    axs[1].set_xlim([-x_max, x_max])
    axs[1].set_ylim([-x_max, x_max])
    cbar2 = fig.colorbar(c2, ax=axs[1], orientation='vertical')
    cbar2.set_label('Value (SymLog scale)', fontsize=fs)

    # Use scientific notation for the colorbar ticks
    cbar2.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar2.formatter.set_powerlimits((-2, 2))
    cbar2.update_ticks()

    # Display plot
    plt.tight_layout()
    plt.savefig("./data/Figure5.png")
    plt.show()

def figure6():
    # Create grid data
    x1_vals = np.linspace(-x_max, x_max, 400)
    x2_vals = np.linspace(-x_max, x_max, 400)
    X1, X2 = np.meshgrid(x1_vals, x2_vals)

    # Compute values of r_1 and \tilde{r}_1
    Z_r1 = r_1m(X1, X2)
    Z_r_tilde_1 = r_1(X1, X2)

    # Create figure
    fig, axs = plt.subplots(1, 1, figsize=(7, 6))

    # Adjust linthresh for more detail around zero
    linthresh = 10  # Increase linthresh to have more detail around zero
    divide = 200

    # Normalize color scale using SymLogNorm for each plot
    norm_r1 = colors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=Z_r1.min(), vmax=Z_r1.max(), base=10)
  
   
    # Plot r_1(x)
    c1 = axs.contourf(X1, X2, Z_r1, levels=divide, cmap='coolwarm', norm=norm_r1)
    axs.set_title(r'$r_1^\diamond(x)$', fontsize=fs)
    axs.set_xlabel('$x_1$', fontsize=fs)
    axs.set_ylabel('$x_2$', fontsize=fs)
    axs.set_xlim([-x_max, x_max])
    axs.set_ylim([-x_max, x_max])
    cbar1 = fig.colorbar(c1, ax=axs, orientation='vertical')
    cbar1.set_label('Value (SymLog scale)', fontsize=fs)

    # Use scientific notation for the colorbar ticks
    cbar1.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar1.formatter.set_powerlimits((-2, 2))
    cbar1.update_ticks()

        # Plot the circle on the first subplot
    circle = plt.Circle((0, 0), R, color='black', linestyle='--', fill=False, lw=2)
    axs.add_patch(circle)

    # Display plot
    plt.tight_layout()
    plt.savefig("./data/Figure6.png")
    plt.show()

if __name__ == "__main__":  
    figure5()
    figure6()