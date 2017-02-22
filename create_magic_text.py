import json

file_path = "/Users/maxwoolf/Downloads/AllCards.json"

separators = {
    'pre': "[",
    'name_manaCost': "@",
    'manaCost_cardtype': "#",
    'cardtype_text': "$",
    'text_power': '%',
    'power_toughness': '^',
    'end': "]"
}
with open('magic_cards.txt', 'wb') as f:
    with open(file_path, 'rb') as data:
        cards = json.load(data)
        names = cards.keys()

        for name in names:
            card = cards[name]
            if not isinstance(card['name'], list):
                manaCost = card.get('manaCost', '')
                cardtype = card.get('type', '')
                text = card.get('text', '').replace(name, "~").replace("\n", "|")
                power_toughness_string = ''

                if 'power' in card:
                    power_toughness_string = separators['text_power'] + \
                        card['power'] + separators['power_toughness'] + \
                        card['toughness']

                card_processed = (separators['pre'] + name + separators['name_manaCost'] +
                                  manaCost +
                                  separators['manaCost_cardtype'] + cardtype +
                                  separators['cardtype_text'] +
                                  text + power_toughness_string + separators['end'] +
                                  "\n")

                f.write(card_processed.encode('utf-8'))
