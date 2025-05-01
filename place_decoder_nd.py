import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

class PlaceFieldSystemBase(ABC):
    def __init__(self, n_cells_per_dim, pos_range=(-1, 1)):
        """Initialize base place field system.
        
        Args:
            n_cells_per_dim: Number of cells along each dimension
            pos_range: (min, max) range of positions in each dimension
        """
        self.n_cells_per_dim = n_cells_per_dim
        self.pos_range = pos_range
        self.centers = self._generate_centers()
        # Set width of Gaussian response (adjust this to control overlap)
        self.sigma = (pos_range[1] - pos_range[0]) / (n_cells_per_dim * 0.5)
        
        # Initialize decoder weights (will be trained later)
        self.weights = torch.nn.Parameter(torch.randn(self.total_cells, self.n_dims))
    
    @property
    @abstractmethod
    def n_dims(self):
        """Number of dimensions for this system."""
        pass
    
    @property
    def total_cells(self):
        """Total number of place cells."""
        return self.n_cells_per_dim ** self.n_dims
    
    @abstractmethod
    def _generate_centers(self):
        """Generate centers for place cells."""
        pass
    
    def compute_place_field(self, position):
        """Compute the response of all place cells for a given position."""
        position = np.array(position)
        # Compute distances to all centers
        distances = np.sqrt(np.sum((self.centers - position) ** 2, axis=1))
        # Compute Gaussian response
        responses = np.exp(-0.5 * (distances / self.sigma) ** 2)
        return responses
    
    def decode_position(self, place_field):
        """Decode position from place cell responses."""
        place_field_tensor = torch.tensor(place_field, dtype=torch.float32)
        # Weighted sum for each dimension
        return torch.matmul(place_field_tensor, self.weights)
    
    @abstractmethod
    def plot_place_fields(self):
        """Plot the response curves of place cells."""
        pass
    
    @abstractmethod
    def plot_population_response(self, position):
        """Plot the population response for a specific position."""
        pass

class PlaceFieldSystem1D(PlaceFieldSystemBase):
    @property
    def n_dims(self):
        return 1
    
    def _generate_centers(self):
        """Generate 1D grid of place cell centers."""
        centers = np.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim)
        return centers.reshape(-1, 1)
    
    def plot_place_fields(self):
        """Plot place fields for 1D case."""
        positions = np.linspace(self.pos_range[0], self.pos_range[1], 200)
        responses = np.array([self.compute_place_field([pos]) for pos in positions])
        
        plt.figure(figsize=(10, 5))
        plt.plot(positions, responses)
        plt.title('1D Place Cell Response Curves')
        plt.xlabel('Position')
        plt.ylabel('Response')
        plt.grid(True)
        plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 1D case."""
        responses = self.compute_place_field([position])
        
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(self.total_cells), responses, alpha=0.6)
        plt.axvline(x=(position - self.pos_range[0]) / 
                   (self.pos_range[1] - self.pos_range[0]) * self.total_cells,
                   color='r', linestyle='--', label=f'True Position: {position:.2f}')
        plt.title(f'1D Population Response at Position {position:.2f}')
        plt.xlabel('Cell Index')
        plt.ylabel('Response')
        plt.grid(True)
        plt.legend()
        plt.show()

class PlaceFieldSystem2D(PlaceFieldSystemBase):
    @property
    def n_dims(self):
        return 2
    
    def _generate_centers(self):
        """Generate 2D grid of place cell centers."""
        x = np.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim)
        y = np.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim)
        xx, yy = np.meshgrid(x, y)
        return np.column_stack((xx.ravel(), yy.ravel()))
    
    def plot_place_fields(self):
        """Plot example place fields for 2D case."""
        x = np.linspace(self.pos_range[0], self.pos_range[1], 50)
        y = np.linspace(self.pos_range[0], self.pos_range[1], 50)
        xx, yy = np.meshgrid(x, y)
        
        # Plot responses for a few example cells
        fig = plt.figure(figsize=(15, 5))
        example_cells = [0, self.total_cells // 4, self.total_cells // 2]
        
        for idx, cell_idx in enumerate(example_cells):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            responses = np.array([self.compute_place_field([x, y])[cell_idx] 
                                for x, y in zip(xx.ravel(), yy.ravel())])
            responses = responses.reshape(xx.shape)
            
            surf = ax.plot_surface(xx, yy, responses, cmap='viridis')
            plt.colorbar(surf, ax=ax)
            ax.set_title(f'2D Place Cell {cell_idx}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_zlabel('Response')
        
        plt.tight_layout()
        plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 2D case."""
        responses = self.compute_place_field(position)
        response_grid = responses.reshape(self.n_cells_per_dim, self.n_cells_per_dim)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(response_grid, extent=[self.pos_range[0], self.pos_range[1],
                                        self.pos_range[0], self.pos_range[1]],
                  origin='lower', cmap='viridis')
        plt.plot(position[0], position[1], 'r*', markersize=15, 
                label=f'True Position: ({position[0]:.2f}, {position[1]:.2f})')
        plt.colorbar(label='Response')
        plt.title('2D Population Response')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.show()

class PlaceFieldSystem3D(PlaceFieldSystemBase):
    @property
    def n_dims(self):
        return 3
    
    def _generate_centers(self):
        """Generate 3D grid of place cell centers."""
        x = np.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim)
        y = np.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim)
        z = np.linspace(self.pos_range[0], self.pos_range[1], self.n_cells_per_dim)
        xx, yy, zz = np.meshgrid(x, y, z)
        return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    
    def plot_place_fields(self):
        """Plot example place fields for 3D case using slices."""
        x = np.linspace(self.pos_range[0], self.pos_range[1], 20)
        y = np.linspace(self.pos_range[0], self.pos_range[1], 20)
        xx, yy = np.meshgrid(x, y)
        
        # Plot responses for example cells at different z-slices
        fig = plt.figure(figsize=(15, 5))
        example_cell = self.total_cells // 2
        z_slices = [-0.5, 0.0, 0.5]
        
        for idx, z in enumerate(z_slices):
            ax = fig.add_subplot(1, 3, idx + 1)
            responses = np.array([self.compute_place_field([x, y, z])[example_cell] 
                                for x, y in zip(xx.ravel(), yy.ravel())])
            responses = responses.reshape(xx.shape)
            
            im = ax.imshow(responses, extent=[self.pos_range[0], self.pos_range[1],
                                            self.pos_range[0], self.pos_range[1]],
                          origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'3D Place Cell at z={z:.1f}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
        
        plt.tight_layout()
        plt.show()
    
    def plot_population_response(self, position):
        """Plot population response for 3D case using orthogonal slices."""
        responses = self.compute_place_field(position)
        response_cube = responses.reshape(self.n_cells_per_dim, 
                                       self.n_cells_per_dim,
                                       self.n_cells_per_dim)
        
        # Plot three orthogonal slices through the true position
        fig = plt.figure(figsize=(15, 5))
        slice_names = ['YZ', 'XZ', 'XY']
        slices = [
            response_cube[self.n_cells_per_dim//2, :, :],
            response_cube[:, self.n_cells_per_dim//2, :],
            response_cube[:, :, self.n_cells_per_dim//2]
        ]
        
        for idx, (slice_data, name) in enumerate(zip(slices, slice_names)):
            ax = fig.add_subplot(1, 3, idx + 1)
            im = ax.imshow(slice_data, origin='lower', cmap='viridis')
            plt.colorbar(im, ax=ax)
            ax.set_title(f'3D Population Response ({name} plane)')
        
        plt.suptitle(f'Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})')
        plt.tight_layout()
        plt.show()

def train_decoder(system, n_epochs=1000, batch_size=32, learning_rate=0.01, val_size=100):
    """Train the decoder weights using random positions."""
    optimizer = torch.optim.Adam([system.weights], lr=learning_rate)
    losses = []
    accuracies = []
    
    for epoch in range(n_epochs):
        # Generate random positions for training
        positions = np.random.uniform(system.pos_range[0], 
                                    system.pos_range[1], 
                                    (batch_size, system.n_dims))
        
        # Compute place fields
        place_fields = np.array([system.compute_place_field(pos) 
                               for pos in positions])
        place_fields_tensor = torch.tensor(place_fields, dtype=torch.float32)
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        
        # Forward pass
        decoded_positions = torch.matmul(place_fields_tensor, system.weights)
        
        # Compute loss (MSE)
        loss = torch.mean(torch.sum((decoded_positions - positions_tensor) ** 2, dim=1))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Compute validation accuracy
        val_positions = np.random.uniform(system.pos_range[0], 
                                        system.pos_range[1], 
                                        (val_size, system.n_dims))
        val_place_fields = np.array([system.compute_place_field(pos) 
                                   for pos in val_positions])
        val_decoded = np.array([system.decode_position(pf).detach().numpy() 
                              for pf in val_place_fields])
        errors = np.sqrt(np.sum((val_decoded - val_positions) ** 2, axis=1))
        mean_error = np.mean(errors)
        accuracies.append(mean_error)
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}, Mean Error: {mean_error:.4f}')
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    
    # Loss curve
    ax1.semilogy(losses)
    ax1.set_title('Training Loss (Log Scale)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (log)')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.grid(True, which="major", ls="-", alpha=0.5)
    
    # Accuracy curve
    ax2.plot(accuracies)
    ax2.set_title('Decoding Error')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Euclidean Error')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return losses, accuracies

def evaluate_decoder(system, n_test=1000):
    """Evaluate decoder accuracy with test positions."""
    # Generate test positions
    test_positions = np.random.uniform(system.pos_range[0], 
                                     system.pos_range[1], 
                                     (n_test, system.n_dims))
    
    # Get decoded positions
    decoded_positions = []
    for pos in test_positions:
        place_field = system.compute_place_field(pos)
        decoded_pos = system.decode_position(place_field).detach().numpy()
        decoded_positions.append(decoded_pos)
    
    decoded_positions = np.array(decoded_positions)
    errors = np.sqrt(np.sum((decoded_positions - test_positions) ** 2, axis=1))
    
    print(f"\nAccuracy Statistics ({system.n_dims}D):")
    print(f"Mean Error: {np.mean(errors):.4f}")
    print(f"Standard Deviation: {np.std(errors):.4f}")
    print(f"Median Error: {np.median(errors):.4f}")
    print(f"90th Percentile Error: {np.percentile(errors, 90):.4f}")
    
    return errors

def main():
    # Test all three systems
    systems = [
        PlaceFieldSystem1D(n_cells_per_dim=50),
        PlaceFieldSystem2D(n_cells_per_dim=10),
        PlaceFieldSystem3D(n_cells_per_dim=7)
    ]
    
    for system in systems:
        print(f"\nTesting {system.n_dims}D system:")
        print(f"Total place cells: {system.total_cells}")
        
        # Show initial place fields
        print("\nPlace field visualization:")
        system.plot_place_fields()
        
        # Train the decoder
        print("\nTraining decoder...")
        train_decoder(system)
        
        # Show example responses
        if system.n_dims == 1:
            positions = [-0.5, 0.0, 0.5]
        elif system.n_dims == 2:
            positions = [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5]]
        else:  # 3D
            positions = [[-0.5, -0.5, -0.5], [0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        
        print("\nExample population responses:")
        for pos in positions:
            system.plot_population_response(pos)
        
        # Evaluate decoder
        evaluate_decoder(system)

if __name__ == "__main__":
    main()
