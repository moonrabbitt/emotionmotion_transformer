# Prototype 10

## Week 18

## Visuals
This week, I ventured into enhancing the visual aspect of my project. I hand-drew sprites and developed functions using Pyglet for displaying these sprites, integrating them with GLSL for dynamic backgrounds and foregrounds. A significant portion of my time was dedicated to resolving a memory leak issue stemming from these new visuals.

The next step involved deploying the chat sentiment analysis, motion generation model, and the newly added visual sprites in parallel. This approach was crucial for minimising the time lag in broadcasting the output. However, achieving smooth communication between these parallel processes without overwhelming my not-so-powerful home computer's memory was a considerable challenge. It required a lot of work to ensure efficient use of memory and synchronisation among the processes, especially since I aimed to keep redundant memory usage to a minimum.

Despite some persistent delay issues with OBS, I managed to get everything to work together quite well when tested in deployment. I'm planning to conduct a user test with a friend later this week.