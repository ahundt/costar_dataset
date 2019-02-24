import os
import datetime
import argparse
from tqdm import tqdm

try:
    import internetarchive
except ImportError:
    raise ImportError('To use this script, you need to install package "internetarchive".')


def _parse_args():
    parser = argparse.ArgumentParser(
            description='CoSTAR BSD File Integrity Check: '
                        'Checks local file hashes against the hash on the Internet Archive. '
                        'This script has an additional dependency on package "internetarchive".')
    parser.add_argument('-p', '--path', type=str, metavar='DIR',
                        default='~/.keras/datasets/costar_block_stacking_dataset_v0.4',
                        help='Full path to the top folder of the dataset. `~` symbol will be expanded. '
                             'Defaults to "~/.keras/datasets/costar_block_stacking_dataset_v0.4"')
    parser.add_argument("--include-ext", type=str, nargs='+', default=['.txt', '.h5f', '.csv', '.yaml'],
                        help='File extensions to check. Default is .txt, .h5f, .csv, and .yaml')
    return parser.parse_args()


def timeStamped(fname, fmt='%Y-%m-%d-%H-%M-%S_{fname}'):
    """ Apply a timestamp to the front of a filename description.

    see: http://stackoverflow.com/a/5215012/99379
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def dump_to_txt(filename, li):
    with open(filename, 'w') as f:
        f.write('\n'.join(li) + '\n')
    print("Output file saved as {}".format(filename))


def main(args):
    # Get the relative path to all matching files in the directory and subdirectories
    args.path = os.path.expanduser(args.path)
    local_filenames = []
    print('Selecting files with extensions: \n{}'.format(args.include_ext))
    for root, _, files in os.walk(args.path):
        for filename in files:
            if any([ext in filename for ext in args.include_ext]):  # filename matches one of the exts
                # Get the relative path to this file
                rel_dir = os.path.relpath(root, args.path)
                if rel_dir == '.':
                    rel_dir = ''
                local_filenames.append(os.path.join(rel_dir, filename))
    if len(local_filenames) == 0:
        raise RuntimeError('No matching files found! '
                           'Are you sure the path is correct? {}'.format(args.path))
    print('Counted {} matching files.'.format(len(local_filenames)))

    # Get metadata for the item from the internetarchive
    item = internetarchive.get_item('johns_hopkins_costar_dataset', debug=True)

    # Get the files hashes on server
    remote_filename_hash = {}
    for server_file in item.files:
        remote_filename_hash[server_file['name']] = server_file['md5']

    # Check the local files for integrity
    progress_bar = tqdm(local_filenames, ascii=True, desc='Hash check in progress')
    not_on_server, hash_mismatch = [], []
    for relative_file_path in progress_bar:
        if relative_file_path not in remote_filename_hash:  # File is not on the server
            not_on_server.append(relative_file_path)
            progress_bar.write('File not on server: {}'.format(relative_file_path))
            continue

        with open(os.path.join(args.path, relative_file_path), 'rb') as f:
            local_md5 = internetarchive.utils.get_md5(f)
            if local_md5 != remote_filename_hash[relative_file_path]:  # Hash mismatch
                hash_mismatch.append(relative_file_path)
                progress_bar.write('File hash mismatch: {}'.format(relative_file_path))

    print("Found {} files not on server: {}".format(len(not_on_server), not_on_server))
    print("Found {} files with hash inconsistent with the server: {}".format(len(hash_mismatch), hash_mismatch))

    # Save the missing/corrupted filenames
    if len(not_on_server) > 0:
        dump_to_txt(timeStamped('integrity_check_not_on_server.txt'), not_on_server)
    if len(hash_mismatch) > 0:
        dump_to_txt(timeStamped('integrity_check_hash_mismatch.txt'), hash_mismatch)


if __name__ == '__main__':
    args = _parse_args()
    main(args)
