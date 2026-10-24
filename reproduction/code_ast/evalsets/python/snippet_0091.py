"""
  " Tests for evaluator.py
  """
import pytest
from ..src import playingcard as pc
from ..src import evaluator as ev
from . import tools


@pytest.fixture
def ace():
    return pc.PlayingCard('A', 's')


def test_iswheel_A234_returnsFalse():
    cards = tools.to_cards(['As', '2c', '3d', '4d'])
    assert ev.is_wheel(cards) is False


def test_iswheel_A2345_returnsTrue():
    cards = tools.to_cards(['As', '2c', '3d', '4d', '5h'])
    assert ev.is_wheel(cards)


def test_iswheel_AA23_returnsFalse():
    cards = tools.to_cards(['As', 'Ah', '2c', '3d'])
    assert ev.is_wheel(cards) is False


def test_dominantsuit_1card_returnssuit():
    cards = [pc.PlayingCard('A', 's')]
    assert ev.dominant_suit(cards) == 's'


def test_dominantsuit_2diffranks_returnshigherrank():
    cards = tools.to_cards(['As', 'Kc'])
    assert ev.dominant_suit(cards) == 's'


def test_dominantsuit_3diffranks_returnshigherrank():
    cards = tools.to_cards(['Kc', 'As', 'Qh'])
    assert ev.dominant_suit(cards) == 's'


def test_dominantsuit_4diffranks_returnshigherrank():
    cards = tools.to_cards(['Kc', 'As', 'Jd', 'Qh'])
    assert ev.dominant_suit(cards) == 's'


def test_dominantsuit_3cards2suitedSpades_returnsSpades():
    cards = tools.to_cards(['Kc', 'As', 'Qs'])
    assert ev.dominant_suit(cards) == 's'


def test_dominantsuit_HigherSpades_returnsSpades():
    cards = tools.to_cards(['Ac', 'Ks', 'As', 'Qc'])
    assert ev.dominant_suit(cards) == 's'


def test_isset_0cards_returnsFalse():
    assert ev.is_set([])


def test_isset_1card_returnsTrue(ace):
    assert ev.is_set([ace])


def test_isset_dupes_returnsFalse(ace):
    assert ev.is_set([ace, ace]) is False


def test_isstraight_lowstraight_returns5():
    hand = tools.make('straight_low')
    assert ev.is_straight(hand) == 5


def test_isstraight_midstraight_returnsPostiveNum():
    hand = tools.make('straight_mid')
    assert ev.is_straight(hand) > 0


def test_isstraight_highstraight_returnsA():
    hand = tools.make('straight_high')
    assert ev.is_straight(hand) == 14


def test_isstraight_nonstraight_returns0():
    hand = tools.make('wheeldraw')
    assert ev.is_straight(hand) == 0


def test_issuited_1card_returnsTrue(ace):
    assert ev.is_suited([ace])


def test_issuited_2suitedcards_returnsTrue():
    cl = tools.to_cards(['As', '2s'])
    assert ev.is_suited(cl)


def test_issuited_2unsuitedcard_returnsFalse():
    cl = tools.to_cards(['As', 'Ad'])
    assert ev.is_suited(cl) is False


def test_isvalidhand_4cards_returnsFalse():
    h = tools.get_cards(4)
    assert ev.is_validhand(h) is False


def test_isvalidhand_5cards_returnsTrue():
    h = tools.get_cards(5)
    assert ev.is_validhand(h)


def test_isvalidhand_6cards_returnsFalse():
    h = tools.get_cards(6)
    assert ev.is_validhand(h) is False


def test_isvalidhand_duplicateCards_returnsFalse():
    h = tools.make('dupes')
    assert ev.is_validhand(h) is False


def test_isvalidhand_royalflush_returnsTrue():
    h = tools.make('royalflush')
    assert ev.is_validhand(h)


def test_findbesthand_pair_returnsPAIR():
    cards = tools.to_cards(['2c', '3c', '5s', '7s', 'Kc', 'Ac', 'As'])
    besthand = ev.find_best_hand(cards)
    val = ev.get_value(besthand)
    assert ev.get_type(val) == 'PAIR'


def test_findbesthand_fullhouse_returnsFULLHOUSE():
    cards = tools.to_cards(['7c', '7s', 'Ks', 'Kc', 'Ah', 'Ac', 'As'])
    besthand = ev.find_best_hand(cards)
    val = ev.get_value(besthand)
    assert ev.get_type(val) == 'FULL HOUSE'


def test_findbesthand_straightflush_returnsSTRAIGHTFLUSH():
    cards = tools.to_cards(['4s', '5s', '6s', '7s', '8s', 'Ks', 'As'])
    besthand = ev.find_best_hand(cards)
    val = ev.get_value(besthand)
    assert ev.get_type(val) == 'STRAIGHT FLUSH'


def test_findbesthand_quads_returnsQUADS():
    cards = tools.to_cards(['Kc', 'Kd', 'Ks', 'Ac', 'Kd', 'Ah', 'As'])
    besthand = ev.find_best_hand(cards)
    val = ev.get_value(besthand)
    assert ev.get_type(val) == 'QUADS'


def test_findbesthand_straight_returnsSTRAIGHT():
    cards = tools.to_cards(['Ac', 'As', '2c', '3s', '4h', '5s', '5h'])
    besthand = ev.find_best_hand(cards)
    val = ev.get_value(besthand)
    assert ev.get_type(val) == 'STRAIGHT'


def test_findbesthand_flush_returnsFLUSH():
    cards = tools.to_cards(['8s', '9s', 'Tc', 'Js', 'Qs', 'Ks', 'Ac'])
    besthand = ev.find_best_hand(cards)
    val = ev.get_value(besthand)
    assert ev.get_type(val) == 'FLUSH'


def test_getallgaps_1card_returns0():
    cards = tools.to_cards(['Kc', 'As'])
    assert ev.get_allgaps(cards) == 0


def test_getallgaps_2connected_returns0():
    cards = tools.to_cards(['Kc', 'As'])
    assert ev.get_allgaps(cards) == 0


def test_getallgaps_2cards1gap_returns1():
    cards = tools.to_cards(['Qc', 'As'])
    assert ev.get_allgaps(cards) == 1


def test_getallgaps_3cards1gap_returns1():
    cards = tools.to_cards(['Tc', 'Js', 'Ks'])
    assert ev.get_allgaps(cards) == 1


def test_getgap_23_returns0():
    c1 = pc.PlayingCard('2', 's')
    c2 = pc.PlayingCard('3', 's')
    assert ev.get_gap(c1, c2) == 0


def test_getgap_32_returns0():
    c1 = pc.PlayingCard('2', 's')
    c2 = pc.PlayingCard('3', 's')
    assert ev.get_gap(c2, c1) == 0


def test_getgap_24_returns1():
    c1 = pc.PlayingCard('2', 's')
    c2 = pc.PlayingCard('4', 's')
    assert ev.get_gap(c2, c1) == 1


def test_getgap_2A_returns11():
    c1 = pc.PlayingCard('2', 's')
    c2 = pc.PlayingCard('A', 's')
    assert ev.get_gap(c1, c2) == 11


def test_getgap_22_returnsNeg1():
    c1 = pc.PlayingCard('2', 's')
    c2 = pc.PlayingCard('2', 'c')
    assert ev.get_gap(c1, c2) == -1


def test_gettype_negativevalue_returnsINVALID():
    assert ev.get_type(-1) == 'INVALID'


def test_gettype_1000000000000_raiseEx():
    with pytest.raises(ValueError):
        ev.get_type(1000000000000)


def test_rankdict_0K_counts0(ace):
    rankdict = ev.rank_dict([ace])
    # 0 is the default in case there are no Aces
    assert rankdict.get('K', 0) == 0


def test_rankdict_1Ace_counts1():
    cl = tools.to_cards(['As', '2s'])
    rankdict = ev.rank_dict(cl)
    assert rankdict.get('A') == 1


def test_rankdict_2Aces_counts2():
    cl = tools.to_cards(['As', 'Ah', 'Kc'])
    rankdict = ev.rank_dict(cl)
    assert rankdict.get('A') == 2


def test_ranklist_1Ace_lenEquals1(ace):
    ranklist = ev.rank_list([ace])
    assert len(ranklist) == 1


def test_ranklist_1Ace_1AceCounted(ace):
    ranklist = ev.rank_list([ace])
    print(ranklist)
    assert ranklist[0][0] == 1
    assert ranklist[0][1] == 'A'


def test_ranklist_2Aces_lenEquals1():
    cl = tools.to_cards(['Ah', 'As'])
    ranklist = ev.rank_list(cl)
    assert len(ranklist) == 1


def test_ranklist_2Aces_2AcesCounted():
    cl = tools.to_cards(['Ah', 'As'])
    ranklist = ev.rank_list(cl)
    assert ranklist[0][0] == 2
    assert ranklist[0][1] == 'A'


def test_ranklist_AK_lenEquals2():
    cl = tools.to_cards(['Kh', 'As'])
    ranklist = ev.rank_list(cl)
    assert len(ranklist) == 2


def test_removepairs_22_returns2():
    cl = tools.to_cards(['2c', '2d'])
    assert ev.remove_pairs(cl).pop().rank == '2'

    # assert cl.remove_pairs() == ['2c']


def test_removepairs_2345_returns2345():
    cards = tools.to_cards(['2c', '3d', '4d', '5h'])
    assert ev.remove_pairs(cards) == cards


def test_removepairs_A223_returns23A():
    # Keep the first 2
    cl = tools.to_cards(['As', '2h', '2c', '3d'])
    expected = tools.to_cards(['2h', '3d', 'As'])
    assert ev.remove_pairs(cl) == expected


def test_scoreranklist_A_return14():
    cards = [pc.PlayingCard('A', 's')]
    expected = 14 * ev.MULTIPLIERS[0]
    rd = ev.rank_list(cards)
    assert ev.score_ranklist(rd) == expected


def test_scorecardlist_A_returns14():
    cards = [pc.PlayingCard('A', 's')]
    assert ev.score_cardlist(cards) == 14


def test_scorecardlist_AK_returns1413():
    cards = tools.to_cards(['As', 'Ks'])
    assert ev.score_cardlist(cards) == 1413


def test_scorecardlist_AKQ_returns141312():
    cards = tools.to_cards(['As', 'Ks', 'Qs'])
    assert ev.score_cardlist(cards) == 141312


def test_scorepairhands_1Ace_returns1400000000():
    cards = [pc.PlayingCard('A', 's')]
    ranklist = ev.rank_list(cards)
    assert ev.score_pair_hands(cards, ranklist) == 1400000000


def test_scorepairhands_2Aces_returns21400000000():
    cards = tools.to_cards(['As', 'Ah'])
    ranklist = ev.rank_list(cards)
    assert ev.score_pair_hands(cards, ranklist) == 21400000000


def test_scorepairhands_3Aces_returns41400000000():
    cards = tools.to_cards(['As', 'Ah', 'Ac'])
    ranklist = ev.rank_list(cards)
    assert ev.score_pair_hands(cards, ranklist) == 41400000000


def test_scorepairhands_4Aces_returns81400000000():
    cards = tools.to_cards(['As', 'Ah', 'Ac', 'Ad'])
    ranklist = ev.rank_list(cards)
    assert ev.score_pair_hands(cards, ranklist) == 81400000000


def test_scorepairhands_AAKK_returns31413000000():
    cards = tools.to_cards(['As', 'Ah', 'Kc', 'Kd'])
    ranklist = ev.rank_list(cards)
    assert ev.score_pair_hands(cards, ranklist) == 31413000000


def test_suitedcarddict_0Spades_listlenEquals0():
    cl = tools.to_cards(['2h', 'Kc'])
    suitdict = ev.suitedcard_dict(cl)
    # Empty list is the default in case there are no Aces
    assert len(suitdict.get('s', [])) == 0


def test_suitedcarddict_1Spade_listlenEquals1():
    cl = tools.to_cards(['As', 'Kc'])
    suitdict = ev.suitedcard_dict(cl)
    # Empty list is the default in case there are no Aces
    assert len(suitdict.get('s', [])) == 1