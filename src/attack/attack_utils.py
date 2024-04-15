class BaseAttack():
    def __init__():
        pass

def prompt_composer(template: str, placeholders: dict):
    for placeholder, value in placeholders.items():
        template = template.replace(placeholder, value)
    return template

def check_bag_of_words(phrase_a: str, phrase_b: str, debug = False) -> bool:
  """Uses bag of words and substring comparison to check if one phrase is
  included in another. Note that this is a symmetric function.

  Args:
    phrase_a: the first phrase to be compared
    phrase_b: the second phrase to be compared
    debug: whether to print debug message

  Example:
    has_knowledge("A, B, and C", "C, B, D, A, and p")
    truth_set: {'', 'B', 'C', 'A', 'and'}
    answer_set: {'', 'D', 'B', 'C', 'A', 'p', 'and'}
    => TRUE
  """
  set_a = set(re.split("\s|(?<!\d)[,.;](?!\d)", phrase_a))
  set_b = set(re.split("\s|(?<!\d)[,.;](?!\d)", phrase_b))
  if debug:
    print("phrase_a:", phrase_a)
    print("phrase_b:", phrase_b)
    print("set_a:", set_a)
    print("set_b:", set_b)
  return set_a.issubset(set_b) or set_b.issubset(set_a)