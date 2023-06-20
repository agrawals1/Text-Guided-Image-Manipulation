import os
from utils.inst_optim import StyleGANInverter
from utils.visualizer import save_image, load_image, resize_image


def main(model='styleganinv_ffhq256', text='he is old', lr=.01, iterations=500, clip_loss_wt=2.0,
        image_path="examples/test.jpg", loss_weight_feat=5e-5, loss_weight_enc = 2.0, mode='man', num_results=5):
  os.environ["CUDA_VISIBLE_DEVICES"] = '0'
  assert os.path.isfile(image_path)
  output_dir = 'results/inversion/test'
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  inverter = StyleGANInverter(
      model,
      mode = mode,                         
      learning_rate=lr,
      iteration=iterations,
      reconstruction_loss_weight=1.0,
      perceptual_loss_weight=loss_weight_feat,
      regularization_loss_weight=loss_weight_enc,
      clip_loss_weight=clip_loss_wt,
      description=text)
  
  image_size = inverter.G.resolution

  # Image inversion code
  img = resize_image(load_image(image_path), (image_size, image_size))
  _, results = inverter.invert(img, num_viz=num_results)

  if mode == 'man':
    image_name = os.path.splitext(os.path.basename(image_path))[0]
  else:
    image_name = 'gen'
  save_image(f'{output_dir}/{image_name}_org.png', results[0])
  save_image(f'{output_dir}/{image_name}_encode.png', results[1])
  save_image(f'{output_dir}/{image_name}_invert.png', results[-1])
  

if __name__ == '__main__':
  main()