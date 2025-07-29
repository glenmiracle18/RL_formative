import pygame

def initialize_screen():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Genetic Mutation Environment")
    return screen

def draw_state(screen, evo2_score, budget, mutation_type):
    screen.fill((255, 255, 255))  # White background

    # Draw EVO2 score as a colored bar
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue for mutation types
    bar_width = int(evo2_score * 300)
    pygame.draw.rect(screen, colors[mutation_type], (250, 250, bar_width, 50))

    # Draw budget indicator
    budget_width = int(budget * 300)
    pygame.draw.rect(screen, (0, 0, 0), (250, 350, budget_width, 20))

    pygame.display.flip()