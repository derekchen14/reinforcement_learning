import matplotlib.pyplot as plt

class Artist(object):
  def __init__(self, filename, color='b', show=False, save=False):
    self.show = show
    self.save = save
    self.color = color
    self.filename = filename

  def draw(self, rewards):
    plt.plot(rewards, self.color)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.ylim(0, 400)
    if self.show:
      plt.show()
    if self.save:
      plt.savefig('results/{}.png'.format(self.filename), bbox_inches='tight')
