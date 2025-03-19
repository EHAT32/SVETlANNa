from svetlanna import Clerk
from svetlanna.clerk import ClerkMode
import pytest


def test_init(tmp_path):
    # Test the experiment directory
    clerk = Clerk(tmp_path)
    assert clerk.experiment_directory == tmp_path

    # Test if the experiment directory is not a directory case
    new_path = tmp_path / 'test'
    assert not new_path.exists()
    with open(new_path, 'w'):
        pass

    with pytest.raises(ValueError):
        clerk = Clerk(new_path)


def test_make_experiment_dir(tmp_path):
    new_path = tmp_path / 'test'
    clerk = Clerk(new_path)

    assert not new_path.exists()
    clerk._make_experiment_dir()
    assert new_path.exists()


def test_path_log(tmp_path):
    clerk = Clerk(tmp_path)

    tag = '123'
    assert clerk._path_log(tag) == tmp_path / (tag + '.jsonl')


def test_path_checkpoint(tmp_path):
    clerk = Clerk(tmp_path)

    index = 123
    assert clerk._path_checkpoint(index) == tmp_path / (str(index) + '.pt')

    index = '321.pt'
    assert clerk._path_checkpoint(index) == tmp_path / index


def test_get_log_stream(tmp_path):
    clerk = Clerk(tmp_path)

    tag = '123'
    with clerk._get_log_stream(tag) as stream:
        # Test if the file was created
        assert (tmp_path / (tag + '.jsonl')).exists()

    # Test if the stream is not closed after the context is closed
    assert not stream.closed

    # Test if the same stream is used for the same tag
    with clerk._get_log_stream(tag) as stream2:
        assert stream is stream2

    # Test if the same stream is not used for the different tag
    other_tag = '312'
    assert tag != other_tag
    with clerk._get_log_stream(other_tag) as stream3:
        assert stream is not stream3


def test_get_log_stream_mode(tmp_path):
    clerk = Clerk(tmp_path)

    tag = '123'
    # Test if the stream mode is 'w' for 'new_run' mode
    # By default 'new_run' mode is used
    with clerk:
        with clerk._get_log_stream(tag) as stream:

            assert clerk._mode == ClerkMode.new_run
            assert stream.mode == 'w'

    # Test if the stream mode is 'a' for 'resume' mode
    # The clerk.begin() should be used to set 'resume' mode
    with clerk.begin(resume=True):
        with clerk._get_log_stream(tag) as stream:

            assert clerk._mode == ClerkMode.resume
            assert stream.mode == 'a'


def test_get_log_stream_flushed(tmp_path):
    # TODO: refactoring
    clerk = Clerk(tmp_path)
    tag = '123'

    with clerk._get_log_stream(tag) as stream:
        pass

    # Test if flush does not called after context is closed
    is_flushed = False

    def monkey_flush():
        nonlocal is_flushed
        is_flushed = True

    stream.flush = monkey_flush

    with clerk._get_log_stream(tag) as stream:
        pass
    assert not is_flushed

    # Test if flush is called after context is closed if flush is true
    with clerk._get_log_stream(tag, flush=True) as stream:
        pass
    assert is_flushed


def test_conditions(tmp_path):
    experiment_dir = tmp_path / 'experiment'
    clerk = Clerk(experiment_dir)

    conditions = {
        'test1': 123,
        'test2': [
            123,
            10.,
            'a'
        ],
        'test3': {
            't': 'e',
            's': 't'
        }
    }
    clerk.save_conditions(conditions)

    # Test if the folder and the file are created
    assert experiment_dir.exists()
    assert (experiment_dir / 'conditions.json').exists()

    # Test if when loaded, the conditions are the same
    new_clerk = Clerk(experiment_dir)
    loaded_conditions = new_clerk.load_conditions()

    assert loaded_conditions is not conditions
    assert loaded_conditions == conditions
