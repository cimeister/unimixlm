from scipy.stats import hypergeom

decksize = 60
total_lands_in_deck = 24
opening_hand_size = 7
desired_lands = 5
by_turn = 5
on_play = 1


def CardDrawnProbability(decksize, total_desired_cards, opening_hand_size, desired_cards, by_turn, on_play):
    # hypergeom(total_number_of_objects, total_number_of_successes_available, number_drawn_without_replacement)
    #   .pmf(number_of_successes_desired)

    sum = 0
    for x in range(desired_cards, min(opening_hand_size + by_turn - on_play, total_desired_cards)):
        sum += hypergeom(decksize, total_desired_cards, opening_hand_size + by_turn - on_play).pmf(x)

    return sum

def TurnHighestProbabilityToDraw(decksize, total_desired_cards, opening_hand_size, desired_cards, on_play):
    turn = 1
    pmax = 0
    for c in range(1, decksize):
        p = CardDrawnProbability(decksize, total_desired_cards, opening_hand_size, desired_cards, c, on_play)
        if p > pmax:
            pmax = p
            turn = c

    return turn, pmax

def GoodAdventureOpenerIndatha(decksize, swamp_sources, forest_sources, plains_sources, colorless_sources, on_play):
    opening_hand_size = 7

    # 4x edgewall, 4x foulmire, 4x shepherd, 4x lovestruck

    # edgewall + foulmire
    p_1s = CardDrawnProbability(decksize, swamp_sources, 7, 1, 2, on_play)
    p_1g = CardDrawnProbability(decksize, forest_sources, 7, 1, 2, on_play)
    p_ei = CardDrawnProbability(decksize, 4, 7, 1, 2, on_play)
    p_fk = CardDrawnProbability(decksize, 4, 7, 1, 2, on_play)
    p_a1 = p_1s * p_1g * p_ei * p_fk
    print("edgewall + foulmire: ", p_a1)

    # lovestruck
    p_1g = CardDrawnProbability(decksize, forest_sources, 7, 1, 2, on_play)
    p_2c = CardDrawnProbability(decksize - 1, colorless_sources - 1, 7, 2, 3, on_play)
    p_ls = CardDrawnProbability(decksize, 4, 7, 1, 3, on_play)
    p_a2 = p_1g * p_2c * p_ls
    print("lovestruck: ", p_a2)

    return p_a1 + p_a2
    
def GoodOpenerEerionAdventures(decksize, swamp_sources, forest_sources, plains_sources, colorless_sources, on_play):
    p_adv = GoodAdventureOpenerIndatha(decksize, swamp_sources, forest_sources, plains_sources, colorless_sources, on_play)

    # 2x birth of meletis
    p_1p = CardDrawnProbability(decksize, plains_sources, 7, 1, 2, on_play)
    p_1c = CardDrawnProbability(decksize - 1, colorless_sources - 1, 7, 1, 2, on_play)
    p_bm = CardDrawnProbability(decksize, 2, 7, 1, 2, on_play)
    p_birth = p_1p * p_1c * p_bm
    print("birth: ", p_birth)
        
    # 4x llanowar elves
    p_1g = CardDrawnProbability(decksize, forest_sources, 7, 1, 3, on_play)
    p_2c = CardDrawnProbability(decksize - 1, colorless_sources - 1, 7, 2, 3, on_play)
    p_1l = CardDrawnProbability(decksize, 4, 7, 1, 3, on_play)
    p_llano = p_1g * p_2c * p_1l
    print("llano: ", p_llano)

    # 4x mire triton
    p_1b = CardDrawnProbability(decksize, swamp_sources, 7, 1, 2, on_play)
    p_1c = CardDrawnProbability(decksize - 1, colorless_sources - 1, 7, 1, 2, on_play)
    p_mt = CardDrawnProbability(80, 4, 7, 1, 2, 1)
    p_mire = p_1b * p_1c * p_mt
    print("mire: ", p_mire)

    return p_adv + p_birth + p_llano + p_mire


##probs = CardDrawnProbability(decksize, total_lands_in_deck, opening_hand_size, desired_lands, by_turn, on_play)
##print(probs)
##
##multi = 0
###plains = CardDrawnProbability(80, 12, 7, 2, 7, 1) # eerie ultimatum plains
##plains = CardDrawnProbability(80, 14, 7, 2, 7, 1) # eerie ultimatum plains
###swamp = CardDrawnProbability(80, 11, 7, 3, 7, 1) # eerie ultimatum swamp
##swamp = CardDrawnProbability(80, 15, 7, 3, 7, 1) # eerie ultimatum swamp
###forest = CardDrawnProbability(80, 13, 7, 2, 7, 1) # eerie ultimatum forest
##forest = CardDrawnProbability(80, 17, 7, 2, 7, 1) # eerie ultimatum forest
##ult = CardDrawnProbability(80, 3, 7, 1, 7, 1) # eerie ultimatum
##
##print("plains: " + str(plains))
##print("swamp: " + str(swamp))
##print("forest: " + str(forest))
##print("ult: " + str(ult))
##
##multi = plains * swamp * forest * ult
##print("all: " + str(multi))
##
##print("")

##eerie_turn, e_p = TurnHighestProbabilityToDraw(80, 4, 7, 1, 1)
##print("eerie turn: ", eerie_turn)
##print("eerie prob: ", e_p)
##swamp_turn, s_p = TurnHighestProbabilityToDraw(80, 15, 7, 3, 1)
##print("swamp turn: ", swamp_turn)
##print("swamp prob: ", s_p)

# s = 10, p = 6, f = 6, fp = 4, sf = 2, sp = 2
turn = 10
swamps = 18#13#18
plains = 12#11#12
forests = 16#17#16
colorless = 30
eeries = 3
print("turn ", turn)
eerie_prob = CardDrawnProbability(80, eeries, 7, 1, turn, 1)
print("eerie prob on turn: ", eerie_prob)
swamp_prob = CardDrawnProbability(80, swamps, 7, 3, turn, 1)
print("swamp prob on turn: ", swamp_prob)
plain_prob = CardDrawnProbability(80, plains, 7, 2, turn, 1)
print("plain prob on turn: ", plain_prob)
forest_prob = CardDrawnProbability(80, forests, 7, 2, turn, 1)
print("forest prob on turn: ", forest_prob)

ult_prob = eerie_prob * swamp_prob * plain_prob * forest_prob
print("eerie ultimatum on turn: ", ult_prob)


mire_prob = CardDrawnProbability(80, 4, 7, 1, turn, 1)
print("mire prob on turn: ", mire_prob)
ult_mire_prob = ult_prob * mire_prob
print("mire + eerie on turn: ", ult_mire_prob)

p_ad = GoodAdventureOpenerIndatha(80, swamps, forests, plains, colorless, 1)
print("adventure opener: ", p_ad)

p_eerie_open = GoodOpenerEerionAdventures(80, swamps, forests, plains, colorless, 1)
print("eerie good opener: ", p_eerie_open)