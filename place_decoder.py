import numpy as np
import torch
import matplotlib.pyplot as plt

class PlaceFieldSystem:
    def __init__(self, n_cells=50, pos_range=(-1, 1)):
        """Initialize the place field system.
        
        Args:
            n_cells: Number of place cells
            pos_range: (min, max) range of positions
        """
        self.n_cells = n_cells
        self.pos_range = pos_range
        # Generate evenly spaced centers for place cells
        self.centers = np.linspace(pos_range[0], pos_range[1], n_cells)
        # Set width of Gaussian response (adjust this to control overlap)
        self.sigma = (pos_range[1] - pos_range[0]) / (n_cells * 0.5)
        
        # Initialize decoder weights (will be trained later)
        self.weights = torch.nn.Parameter(torch.randn(n_cells))
        
    def compute_place_field(self, position):
        """Compute the response of all place cells for a given position.
        
        Args:
            position: scalar position value
            
        Returns:
            numpy array of shape (n_cells,) containing cell responses
        """
        # Compute Gaussian response for each cell
        responses = np.exp(-0.5 * ((position - self.centers) / self.sigma) ** 2)
        return responses
    
    def decode_position(self, place_field):
        """Decode position from place cell responses using learned weights.
        
        Args:
            place_field: numpy array of shape (n_cells,) containing cell responses
            
        Returns:
            scalar decoded position
        """
        # Convert to tensor for PyTorch operations
        place_field_tensor = torch.tensor(place_field, dtype=torch.float32)
        # Linear weighted sum
        return torch.sum(self.weights * place_field_tensor)
    
    def plot_place_fields(self):
        """Plot the response curves of all place cells."""
        positions = np.linspace(self.pos_range[0], self.pos_range[1], 200)
        responses = np.array([self.compute_place_field(pos) for pos in positions])
        
        plt.figure(figsize=(10, 5))
        plt.plot(positions, responses)
        plt.title('Place Cell Response Curves')
        plt.xlabel('Position')
        plt.ylabel('Response')
        plt.grid(True)
        plt.show()
    
    def plot_population_response(self, position):
        """Plot the population response for a specific position."""
        responses = self.compute_place_field(position)
        
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(self.n_cells), responses, alpha=0.6)
        plt.axvline(x=(position - self.pos_range[0]) / (self.pos_range[1] - self.pos_range[0]) * self.n_cells,
                   color='r', linestyle='--', label=f'True Position: {position:.2f}')
        plt.title(f'Population Response at Position {position:.2f}')
        plt.xlabel('Cell Index')
        plt.ylabel('Response')
        plt.grid(True)
        plt.legend()
        plt.show()
    
    def plot_decoding_heatmap(self, n_positions=50):
        """Create a heatmap showing decoding accuracy across positions."""
        test_positions = np.linspace(self.pos_range[0], self.pos_range[1], n_positions)
        place_fields = np.array([self.compute_place_field(pos) for pos in test_positions])
        decoded_positions = np.array([self.decode_position(pf).detach().numpy() for pf in place_fields])
        
        plt.figure(figsize=(10, 4))
        plt.plot(test_positions, decoded_positions, 'b-', label='Decoded')
        plt.plot(test_positions, test_positions, 'r--', label='True')
        plt.fill_between(test_positions, decoded_positions, test_positions, alpha=0.3)
        plt.title('Position Decoding Accuracy')
        plt.xlabel('True Position')
        plt.ylabel('Decoded Position')
        plt.grid(True)
        plt.legend()
        plt.show()
        
    def plot_example_decoding(self, n_samples=10):
        """Plot true vs decoded positions for random samples."""
        # Generate random positions
        true_positions = np.random.uniform(self.pos_range[0], 
                                         self.pos_range[1], 
                                         n_samples)
        
        # Get decoded positions
        decoded_positions = []
        for pos in true_positions:
            place_field = self.compute_place_field(pos)
            decoded_pos = self.decode_position(place_field).detach().numpy()
            decoded_positions.append(decoded_pos)
            
        plt.figure(figsize=(8, 8))
        plt.scatter(true_positions, decoded_positions, alpha=0.6)
        plt.plot([self.pos_range[0], self.pos_range[1]], 
                [self.pos_range[0], self.pos_range[1]], 
                'r--', label='Perfect decoding')
        plt.title('True vs Decoded Positions')
        plt.xlabel('True Position')
        plt.ylabel('Decoded Position')
        plt.grid(True)
        plt.legend()
        plt.show()

def train_decoder(system, n_epochs=1000, batch_size=32, learning_rate=0.01):
    """Train the decoder weights using random positions."""
    optimizer = torch.optim.Adam([system.weights], lr=learning_rate)
    losses = []
    
    for epoch in range(n_epochs):
        # Generate random positions for training
        positions = np.random.uniform(system.pos_range[0], 
                                    system.pos_range[1], 
                                    batch_size)
        
        # Compute place fields and convert to tensor
        place_fields = np.array([system.compute_place_field(pos) 
                               for pos in positions])
        place_fields_tensor = torch.tensor(place_fields, dtype=torch.float32)
        positions_tensor = torch.tensor(positions, dtype=torch.float32)
        
        # Forward pass
        decoded_positions = torch.sum(system.weights * place_fields_tensor, dim=1)
        
        # Compute loss (MSE)
        loss = torch.mean((decoded_positions - positions_tensor) ** 2)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.6f}')
    
    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.semilogy(losses)  # Plot with log scale on y-axis
    plt.title('Training Loss (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log)')
    plt.grid(True, which="both", ls="-", alpha=0.2)  # Add grid for both major and minor ticks
    plt.grid(True, which="major", ls="-", alpha=0.5)  # Emphasize major grid
    plt.show()
    
    return losses

# Main execution
if __name__ == "__main__":
    # Create and visualize the system
    system = PlaceFieldSystem()
    print("Initial place field visualization:")
    system.plot_place_fields()
    
    # Train the decoder
    print("\nTraining decoder...")
    losses = train_decoder(system)
    
    # Show decoding results
    print("\nDecoding results:")
    system.plot_example_decoding()
    print("\nDecoding accuracy across position range:")
    system.plot_decoding_heatmap()
    
    # Show population response for example positions
    print("\nExample population responses with trained decoder:")
    system.plot_population_response(-0.5)
    system.plot_population_response(0.0)
    system.plot_population_response(0.5)
    
    # Calculate and display accuracy statistics
    n_test = 1000
    test_positions = np.random.uniform(system.pos_range[0], 
                                     system.pos_range[1], 
                                     n_test)
    decoded_positions = []
    
    for pos in test_positions:
        place_field = system.compute_place_field(pos)
        decoded_pos = system.decode_position(place_field).detach().numpy()
        decoded_positions.append(decoded_pos)
    
    errors = np.abs(np.array(decoded_positions) - test_positions)
    
    print("\nAccuracy Statistics:")
    print(f"Mean Absolute Error: {np.mean(errors):.4f}")
    print(f"Standard Deviation of Error: {np.std(errors):.4f}")
    print(f"Median Error: {np.median(errors):.4f}")
    print(f"90th Percentile Error: {np.percentile(errors, 90):.4f}")
