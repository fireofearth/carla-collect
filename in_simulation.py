import os
import argparse
import logging

from collect.in_simulation import OnlineManager

def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise argparse.ArgumentTypeError(
                f"readable_dir:{s} is not a valid path")

def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
            description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Show debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    
    args = argparser.parse_args()
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)
    print(__doc__)

    try:
        mgr = OnlineManager(args)
        mgr.run()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':
    main()