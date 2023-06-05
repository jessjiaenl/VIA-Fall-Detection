import pygame

# Initialize Pygame
pygame.init()

# Set up the display window
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Shape Drawing")

# Set colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Game loop
running = True
while running:
    # Clear the screen
    screen.fill(BLACK)
    
    # Draw shapes
    pygame.draw.rect(screen, RED, (100, 100, 200, 150))  # Draw a red rectangle
    pygame.draw.circle(screen, GREEN, (400, 300), 100)  # Draw a green circle
    pygame.draw.line(screen, BLUE, (500, 100), (700, 500), 5)  # Draw a blue line
    
    # Update the display
    pygame.display.flip()
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()