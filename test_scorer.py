# Testing sharpness
# result = sharpness_score('data/raw/aum/DSC03866-6.jpg')

# Testing detect_people
# result = detect_people('data/raw/aum/DSC03866-6.jpg')
# print(result)

from src.scorer import standard_score

result = standard_score('data/raw/aum/DSC03866-6.jpg')
print(result)