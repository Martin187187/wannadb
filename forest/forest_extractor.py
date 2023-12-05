from forest.parse_examples import preprocess
from forest.visitor import RegexInterpreter
from forest.synthesizer import MultiTreeSynthesizer
from forest.utils import conditions_to_str
from forest.configuration import Configuration


def synthesize(valid, invalid, condition_invalid):

    printer = RegexInterpreter()
    dsl, valid, invalid, condition_invalid, captures, type_validation = preprocess(valid, invalid, condition_invalid)
    synthesizer = MultiTreeSynthesizer(valid, invalid, captures, condition_invalid, dsl, None, configuration=Configuration())

    solution = synthesizer.synthesize()
    if solution is not None:
        regex, capturing_groups, capture_conditions = solution
        conditions, condition_captures = capture_conditions
        solution_str = printer.eval(regex, captures=condition_captures)

        # Conditions are taken from condition_invalids, meaning that the syntax matches, but the semantics itself
        # are incorrect. If no conditionally invalid examples are provided, this cannot be gathered!
        if len(conditions) > 0:
            solution_str += ', ' + conditions_to_str(conditions)

        if len(capturing_groups) > 0:
            print(f'Captures:\n  {printer.eval(regex, captures=capturing_groups)}')

        return solution_str
    else:
        print('Solution not found!')
        return None

