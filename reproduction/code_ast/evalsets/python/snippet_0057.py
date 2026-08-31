from random import normalvariate, SystemRandom
import configparser

from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

from decks.deck_generators import make_land_spells_deck, make_sequence_deck
from configure_shuffler import configure_shuffler


config = configparser.ConfigParser()
config.read(['config.ini'])

random = SystemRandom().random


def evaluate_mulligans():
    num_land = config.getint("deck", "land")
    num_spells = config.getint("deck", "spells")

    deck = make_land_spells_deck(num_land, num_spells)

    cards_to_draw = 7
    while cards_to_draw > 0:
        deck = shuffle_deck(deck)

        hand = deck[0:cards_to_draw]
        num_lands = hand.count("L")
        ratio = num_lands / cards_to_draw
        if (2.75 / 7) <= ratio <= (4.25 / 7):
            break

        cards_to_draw -= 1

    return cards_to_draw


def shuffle_deck(deck, shufflers):
    for shuffler in shufflers:
        deck = shuffler(deck, random)

    return deck


def test_mulligans():
    stats = {}
    for i in range(0, 10000):
        cards_to_draw = evaluate_mulligans()
        if cards_to_draw not in stats:
            stats[cards_to_draw] = 1
        else:
            stats[cards_to_draw] += 1

    print(stats)


def test_distribution():
    turns_mean = config.getint("game", "turns_mean")
    cards_seen_sigma = config.getfloat("game", "cards_seen_sigma")
    max_turns = config.getint("game", "max_turns")
    stats = {}

    num_iterations = config.getint("simulation", "num_iterations")
    for i in range(0, num_iterations):
        deck = make_land_spells_deck()
        deck = shuffle_deck(deck)
        num_turns = normalvariate(turns_mean / max_turns, cards_seen_sigma)
        num_turns *= max_turns
        num_turns = round(num_turns)

        # TODO: don't assume that we always keep our starting hand!
        cards_to_draw = 7 + num_turns  # starting hand of 7
        hand = deck[0:cards_to_draw]

        land = hand.count('L')
        spells = hand.count('S')
        if spells > 0:
            ratio = round(land / spells, 1)
        else:
            ratio = 1000

        if ratio not in stats:
            stats[ratio] = 1
        else:
            stats[ratio] += 1

    total = sum(stats.values())
    deck_size = config.getint("deck", "total_cards")
    for i in range(0, deck_size):
        key = i / 10.0
        if key not in stats:
            value = 0
        else:
            value = stats[key]

        percent = round(100.0 * value / total, 1)
        #        print("%s: %s%%" % (key, percent,))
        print("%s%%" % (percent,))


# test_mulligans()
# test_distribution()


def test_card_positions():
    # Create a 2d array to hold each card's final position distribution
    deck_size = config.getint("deck", "total_cards")
    positions = [x[:] for x in [[0] * deck_size] * deck_size]

    num_iterations = config.getint("simulation", "num_iterations")
    print_interval = int(num_iterations / 50)
    for x in range(0, num_iterations):
        if print_interval > 0 and x % print_interval == 0:
            print(round(100 * x / num_iterations, 2), "%")

        deck_size = config.getint("deck", "total_cards")
        deck = make_sequence_deck(deck_size)
        deck = shuffle_deck(deck, configure_shuffler())
        for card in deck:
            positions[card][deck[card]] += 1

    # Create a probability map, vs. the average
    average = num_iterations / deck_size
    probability_skews = [
        [
            (column / average)
            for column in row
            ]
        for row in positions
        ]

    fig = plt.gcf()
    fig.canvas.set_window_title('Position probability skew')

    plt.imshow(probability_skews, cmap='nipy_spectral', interpolation='none', norm=Normalize(0, 2))
    plt.colorbar()
    plt.title('Position probability skew')
    plt.ylabel('Start Position')
    plt.xlabel('End Position')
    plt.show()


test_card_positions()