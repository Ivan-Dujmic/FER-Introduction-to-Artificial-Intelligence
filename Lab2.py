import sys

# Using frozensets so we can hash them (they are immutable)
clauses = set()    # set{ frozenset(str, bool) } -> set{ clause(literal, negation) }
new_clauses = set() # the same structure
clause_indices: dict[frozenset, int] = {}   # For printing
clause_next_index = 1   # For printing
goal = ''

cooking_starting_clauses: list[str] = []  # Contains the starting clauses for cooking with all command modifications

def file_to_clean_lines(file: str) -> list[str]:
    with open(file, 'r') as f:
        lines = []
        for line in f:
            if line.startswith('#'):    # Remove comments
                continue
            if line:
                lines.append(line.strip().lower())
        return lines
    
def parse_clauses(lines: list[str]):
    global clauses
    global new_clauses
    global clause_indices
    global clause_next_index
    global goal

    # Reset global variables
    clauses = set()
    new_clauses = set()
    clause_indices = {}
    clause_next_index = 1
    goal = ''

    for i in range(len(lines)):
        line = lines[i]
        parts = line.split(' ')
        if i == len(lines) - 1:  # Last line has to be negated (it's the goal) [ A | ~B -> ~A & B]
            goal = ' '.join(parts)
            for part in parts:
                if part == 'v':
                    continue
                elif part.startswith('~'):  # Negation operator
                    new_set = set()
                    new_set.add((part[1:].strip(), False))  # Negate the negation -> no negation
                    add_new_clause(frozenset(new_set))
                else:
                    new_set = set()
                    new_set.add((part.strip(), True))  # Negate -> has negation
                    add_new_clause(frozenset(new_set))   
        else:
            clause_tmp = set()  # Temporary set to store literals that will be converted to a frozenset ; sets solve factorization
            for part in parts:
                if part == 'v': # Skip operators
                    continue
                elif part.startswith('~'):  # Negation operator 
                    part = part[1:]
                    clause_tmp.add((part.strip(), True))
                else:
                    clause_tmp.add((part.strip(), False))   # No negation
            add_new_clause(frozenset(clause_tmp))

def clause_to_str(clause: frozenset) -> str:
    result = ''
    for literal in clause:
        if literal[1]:  # Negation
            result += '~' + literal[0] + ' v '
        else:
            result += literal[0] + ' v '
    return result[:-3]  # Remove the last ' v '

def add_new_clause(new_clause: frozenset, c1: frozenset = {}, c2: frozenset = {}) -> bool:
    global clauses
    global new_clauses
    global clause_indices
    global clause_next_index
    global goal
    # If NIL we are done
    if not new_clause:
        clauses.add(new_clause)
        print(str(clause_next_index) + '. NIL (' + str(clause_indices[c1]) + ', ' + str(clause_indices[c2]) + ')')
        clause_indices[new_clause] = clause_next_index
        clause_next_index += 1
        return True
    # TAUTOLOGY CHECK
    for literal1 in new_clause:
        for literal2 in new_clause:
            if literal1[0] == literal2[0] and literal1[1] != literal2[1]:   # A v ~A -> Tautology
                return  # Tautology found, not adding the clause

    # SUBSUMED CHECK
    removed_one = False
    for clause in list(clauses):    # Making a copy list to avoid problems of removing elements from a set while iterating over it
        # New clause is already satisfied by an existing clause -> not adding it
        # Check removed_one first instead of doing an entire comparison of frozensets for optimization
        # If the new clause satisfies at least one existing clause, then there can't be an existing one that satisfies the new clause
        if not removed_one and new_clause >= clause:
            return False
        elif clause >= new_clause:  # Existing clause is already satisfied by the new clause -> remove it and keep searching
            clauses.remove(clause)
            removed_one = True

    clauses.add(new_clause)
    new_clauses.add(new_clause)
    if not c1 and not c2:   # Initial clauses don't have parents
        print(str(clause_next_index) + '. ' + clause_to_str(new_clause))
    else:
        print(str(clause_next_index) + '. ' + clause_to_str(new_clause) + ' (' + str(clause_indices[c1]) + ', ' + str(clause_indices[c2]) + ')')
    clause_indices[new_clause] = clause_next_index
    clause_next_index += 1
    return False

def resolution():
    global clauses
    global new_clauses
    global clause_indices
    global clause_next_index
    global goal
    while new_clauses:  # If there were new clauses added in the last iteration
        print('===============')
        clauses1 = new_clauses.copy()   # Make the copies here because we will be changing both clauses and new_clauses throughout the iterations
        clauses2 = clauses.copy()
        new_clauses = set() # Reset the new clauses set
        for clause1 in clauses1: # Always take one from the new clauses, otherwise we will be checking the same pairs unnecessarily
            for clause2 in clauses2: # The other clause is either a new one or an old one (all are contained in clauses)
                if clause1 == clause2:
                    continue
                else:
                    for literal in clause1:
                        # Two clauses can be resolved into a new one if they have a complementary literal, the new clause is the union of the two clauses minus the two literals
                        complement_literal = (literal[0], not literal[1])
                        if clause2.__contains__(complement_literal):
                            new_clause = clause1.union(clause2) - frozenset([literal, complement_literal])
                            if add_new_clause(new_clause, clause1, clause2):
                                print('===============')
                                print('[CONCLUSION]: ' + goal + ' is true')
                                return
    print('===============')
    print('[CONCLUSION]: ' + goal + ' is unknown')
    return

def cooking(commands: list[str]):
    global cooking_starting_clauses

    for command in commands:
        print('\nUser\'s command: ' + command)
        if command.endswith('?'):   # Query goal
            command = command[:-2]
            lines = cooking_starting_clauses + [command]
            parse_clauses(lines)
            resolution()
        elif command.endswith('+'): # Add clause
            command = command[:-2]
            cooking_starting_clauses.append(command)
        elif command.endswith('-'): # Remove clause
            command = command[:-2]
            if command in cooking_starting_clauses:
                cooking_starting_clauses.remove(command)

def main():
    global clauses
    global new_clauses
    global clause_indices
    global clause_next_index
    global goal
    global cooking_starting_clauses

    if 'resolution' in sys.argv:
        clauses_file_index = sys.argv.index('resolution') + 1
        file = sys.argv[clauses_file_index]
        lines = file_to_clean_lines(file)
        parse_clauses(lines)
        resolution()
       
    if 'cooking' in sys.argv:
        clauses_file_index = sys.argv.index('cooking') + 1
        commands_file_index = clauses_file_index + 1
        cooking_starting_clauses = file_to_clean_lines(sys.argv[clauses_file_index])
        print('Constructed with knowledge:')
        for line in cooking_starting_clauses:
            print(line.strip())
        commands = file_to_clean_lines(sys.argv[commands_file_index])
        cooking(commands)

if __name__ == '__main__':
    main()