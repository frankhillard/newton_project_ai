# newton_project_ai

Is it possible to extrapolate laws of motion from videos ? What if I throw a tennis ball and record it as videos ? could a neuronal network find some newtonian laws ? effet of gravity ? and air resistance ?

The idea is to

- train a first neuronal network model to detect the tennis ball and to track positions of the ball in videos (in data/videos)
- run this first neuronal network model on the videos to generate and save the positions of the ball in separated files (one per video)
- Once the positions of the ball (called trajectories) are saved (in data/trajectories) for our training set videos , we can train a second neuronal network to understand someting about the trajectory (here we compress the information with a VAE Variable Auto Encoder). Once the autoencoder is trained the layer with lowest dimensions would yield the movement information and these values are called latent dimensions. So latent dimensions are deduced from positions of the tennis ball
- find an equation by interpolating from positions and latent dimensions.

  use YOLO lib for detection but re-train the model on my tennis ball images
  use YOLO lib for tracking the tennis ball positions
  use pytorch to train a VAE and be able to get latent dimensions
  use pytorch and pysr to perform symbolic regression with latent dimensions
