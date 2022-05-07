import re

file_path = 'data/wig/components/raw_components.txt'
result_file = 'data/wig/components/components.csv'

regex = '>(.*)<'

if __name__ == '__main__':
    # download index csv
    # download html with index stocks
    # download csv with data for stock
    with open(file_path, 'r') as f:
        with open(result_file, 'w') as result:
            result.write('name,ticker\n')
            idx = 0
            name = ''
            ticker = ''
            for line in f.readlines():
                if idx == 0:
                    name = re.search(regex, line).group()[1:-1]
                if idx == 1:
                    ticker = re.search(regex, line).group()[1:-1]
                    result.write(f'{name},{ticker}\n')

                idx += 1
                idx = idx % 9
