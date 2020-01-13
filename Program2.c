//Code:

//a)

total_cards = 52
no_of_king = int(input('Enter the number of kings: '))


def find_prob(no_of_king):
    king, c, prob = 4, total_cards, 1
    while(no_of_king > 0):
        prob *= king/c
        no_of_king -= 1
        king -= 1
        c -= 1
    return prob


if(no_of_king > 0 and no_of_king <= 4):
    prob = find_prob(no_of_king)
    a = round(prob*100, 6)
    print(
        f'Probability of drawing {no_of_king} kings from the deck is {a}%')
else:
    print(f'Enter valid number between 1-4')

//b)

passed_in_both = int(input('Percentange of students passed in both test: '))
either_test = int(
    input('Percentage of student passed in either of the test: '))

passed_in_remaining = passed_in_both / either_test
print(f'Students passed in first test is {round(passed_in_remaining*100,2)}%')
