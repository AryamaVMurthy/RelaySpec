from experiments.auf_vllm.build_workload_manifests import exposed_texts, content
from experiments.auf_vllm.extend_data_manifest import groupkey


def test_exposure_includes_nested_user_requests_but_not_answer_text():
    value={'runs':[{'problem':'Find 12 apples.'}],
           'messages':[{'role':'user','content':'Count five coins.'},
                       {'role':'assistant','content':'Never use this answer as a prompt.'}],
           'turns':['First user turn','Second user turn'],
           'output_ids':[1,2,3],'solution':'Not a prompt'}
    texts=set(exposed_texts(value))
    assert texts=={'Find 12 apples.','Count five coins.','First user turn','Second user turn'}
    assert groupkey('Find 12 apples.')==groupkey('find 27 apples!')


def test_workload_instructions_preserve_problem_and_chat():
    row={'problem':'Original question'}
    assert content(row,'chat')==row['problem']
    assert content(row,'math').startswith(row['problem']) and '\\boxed{}' in content(row,'math')
    assert content(row,'code').startswith(row['problem']) and 'Python 3' in content(row,'code')
