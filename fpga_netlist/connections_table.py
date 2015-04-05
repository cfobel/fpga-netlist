# coding: utf-8
import sys
from collections import OrderedDict

import pandas as pd
import numpy as np
from cythrust.device_vector.sort import sort_int32
from cythrust.device_vector.count import count_int32_key
from cythrust import DeviceDataFrame
from .CONNECTIONS_TABLE import (block_type_by_block_key, driver_and_sink_type,
                                connection_delay_type)


try:
    profile
except NameError:
    profile = lambda (f): f


INPUT_DRIVER_PIN = 0
LOGIC_DRIVER_PIN = 4
LOGIC_BLOCK = 0
INPUT_BLOCK = 1
OUTPUT_BLOCK = 2
CLOCK_PIN = 5

CONNECTION_CLOCK = 5
CONNECTION_DRIVER = 200
CONNECTION_SINK = 100
CONNECTION_CLOCK_DRIVER = 30

# Connection type = DRIVER_TYPE + 10 * SINK_TYPE
DELAY_IO_TO_IO = INPUT_BLOCK + 10 * OUTPUT_BLOCK
DELAY_FB_TO_FB = LOGIC_BLOCK + 10 * LOGIC_BLOCK
DELAY_IO_TO_FB = INPUT_BLOCK + 10 * LOGIC_BLOCK
DELAY_FB_TO_IO = LOGIC_BLOCK + 10 * OUTPUT_BLOCK


@profile
def get_connections_frame(net_list_name, h5f_netlists_path):
    h5f = pd.HDFStore(str(h5f_netlists_path), 'r')

    netlist_group = getattr(h5f.root.netlists, net_list_name)

    block_keys = netlist_group.connections.cols.block_key[:]
    block_types = netlist_group.block_types[:]

    connections = pd.DataFrame(
        OrderedDict([('net_key', netlist_group.connections.cols.net_key[:]),
                     ('block_key',
                      netlist_group.connections.cols.block_key[:]),
                     ('block_type', block_types[block_keys]),
                     ('pin_key', netlist_group.connections.cols.pin_key[:])]))

    net_labels = netlist_group.net_labels[:]
    block_labels = netlist_group.block_labels[:]

    connections['net_label'] = net_labels[netlist_group.connections
                                          .cols.net_key[:]]
    connections['block_label'] = block_labels[netlist_group.connections
                                              .cols.block_key[:]]
    connections['block_type'] = block_types[netlist_group.connections.cols
                                            .block_key[:]]
    h5f.close()
    return connections


@profile
def populate_connection_frame(connections):
    '''
    Add the following columns to connections `DataFrame`:

      - `type`, One of the following:
       * `CONNECTION_CLOCK`
       * `CONNECTION_CLOCK_DRIVER`
       * `CONNECTION_SINK`
       * `CONNECTION_DRIVER`,
      - `synchronous`, `True` if block-type is clocked, _i.e., an input,
        output, or synchronous logic block.
    '''
    connections['type'] = CONNECTION_SINK
    connections.loc[connections.pin_key == CLOCK_PIN, 'type'] = \
        CONNECTION_CLOCK
    connections.loc[(connections.pin_key == LOGIC_DRIVER_PIN) |
                    ((connections.pin_key == INPUT_DRIVER_PIN) &
                     (connections.block_type == INPUT_BLOCK)), 'type'] = \
        CONNECTION_DRIVER

    clock_nets = set(connections[connections.type ==
                                 CONNECTION_CLOCK].net_key.unique())
    clock_net_records = connections[
        connections.net_key.isin(clock_nets) &
        (connections.type == CONNECTION_DRIVER)]

    TYPE_INDEX = connections.columns.to_native_types().index('type')
    connections.iloc[clock_net_records.index, TYPE_INDEX] = \
        CONNECTION_CLOCK_DRIVER

    connections['synchronous'] = (connections.block_type.isin([INPUT_BLOCK,
                                                               OUTPUT_BLOCK]) |
                                  (connections.pin_key == CLOCK_PIN))

    block_count = connections.block_key.unique().shape[0]
    block_is_sync = np.zeros(block_count, dtype=bool)
    synchronous_blocks = connections.loc[connections.synchronous, 'block_key']
    block_is_sync[synchronous_blocks.as_matrix()] = True
    connections.loc[:, 'synchronous'] = block_is_sync[connections.block_key
                                                      .values]
    connections = connections.drop_duplicates()
    connections = connections.sort(['net_key'])

    block_count = connections.block_key.unique().shape[0]
    net_count = connections.net_key.unique().shape[0]
    synchronous_blocks = (connections[connections.synchronous].block_key
                          .unique())

    return (connections, synchronous_blocks, block_count, net_count)


class ConnectionsTable(object):
    def __init__(self, connections_frame):
        result = populate_connection_frame(connections_frame)
        self.connections = result[0]
        self.synchronous_blocks = result[1]
        self.block_count = result[2]
        self.net_count = result[3]
        self.net_drivers = (self.driver_connections().block_key
                            .unique())
        self.connections['driver_block_key'] = (
            self.net_drivers[self.connections.net_key.values])
        self._append_type_info()
        self.io_count = self.io_block_keys().size
        self.logic_count = self.logic_block_keys().size

        self.block_data = DeviceDataFrame(
            OrderedDict([('key', self.connections.sink_key.values
                          .astype(np.int32)),
                         ('type', self.connections.sink_type.values
                          .astype(np.uint8))]))

        block_type_by_block_key(self.block_data.v['key'],
                                self.block_data.v['type'])
        self.block_data.drop('key')

        temp = DeviceDataFrame({'logic_block_keys':
                                self.connections[self.connections.sink_type ==
                                                 LOGIC_BLOCK].sink_key.values
                                .astype(np.int32)})

        sort_int32(temp.v['logic_block_keys'])
        temp.add('reduced_block_keys', dtype=np.int32)
        temp.add('block_net_counts', dtype=np.int32)

        N = count_int32_key(temp.v['logic_block_keys'],
                            temp.v['reduced_block_keys'],
                            temp.v['block_net_counts'])
        self.block_net_counts = temp.v['block_net_counts'][:N]

        # ### Blocks with only one connection ###
        #
        # Some net-lists _(e.g. `clma`)_ have at least one logic block that is
        # only connected to a single net.  For each such block, either:
        #
        #  - The block has no inputs, so must be a constant-generator.
        #  - The block has no output, so is equivalent to no block at all.
        #
        # In either case, the block should have no impact on timing, so the
        # arrival-time of the block can be set to zero.
        self.single_connection_blocks = \
            temp.v['reduced_block_keys'][:][self.block_net_counts[:N] < 2]

    def filter(self, connection_mask):
        # # `filter` #
        #
        # Filter connections according to specified mask.
        #
        #  1. Reassign net-keys in connection list
        #    a. Partition connections into non-global/global.
        #
        #                                             non-global
        #        0                                    net count
        #        |                                        |
        #        ╔════════════════════════════════════════╗┌──────────────────┐
        #        ║ non-global net keys                    ║│ global net keys  │
        #        ╚════════════════════════════════════════╝└──────────────────┘
        net_key_connections = self.connections.copy()
        net_key_connections['exclude'] = ~connection_mask
        net_key_connections.sort(['exclude', 'net_key'], inplace=True)
        net_key_connections.drop_duplicates('net_key', inplace=True)
        #    b. Reassign net-keys starting from 0 for non-global nets.        .
        #      * Start with new contiguous range of net keys, `[0, net_count]`.
        #
        #                                             non-global            total
        #        0                                    net count           net count
        #        |                                        |                   |
        #        ┌────────────────────────────────────────────────────────────┐
        #        │ new net keys                                               │
        #        └────────────────────────────────────────────────────────────┘
        net_key_connections['new_net_key'] = np.arange(
            net_key_connections['net_key'].shape[0],
            dtype=net_key_connections['net_key'].dtype)
        #      * Sort "new net keys" by the partitioned net key values, or,
        #        equivalently, set index of data frame to original net-keys.
        #      * Result is the mapping from original net keys to new net keys.
        net_key_connections.set_index('net_key', inplace=True)
        #    c. Scatter new net-keys starting from 0 for non-global nets.
        #      * Use permutation copy to update any data structures containing
        #        original net keys with new net keys, using "net net keys" map.
        connections = self.connections.loc[connection_mask].copy()
        connections.loc[:, 'net_key'] = (net_key_connections
                                         .loc[connections['net_key'].values]
                                         ['new_net_key'].values.ravel())
        return connections

    @classmethod
    def from_net_list_name(cls, net_list_name, h5f_netlists_path):
        return cls(get_connections_frame(net_list_name, h5f_netlists_path))

    def _append_type_info(self):
        block_type_data = DeviceDataFrame(
            OrderedDict([
                ('block_key', self.connections.block_key.values
                .astype(np.int32)),
                ('block_type', self.connections.block_type.values
                .astype(np.uint8))]))

        block_type_by_block_key(block_type_data.v['block_key'],
                                block_type_data.v['block_type'])

        block_type_data.drop('block_key')
        block_type_data.add('driver_key', self.connections
                            .driver_block_key.values.astype(np.int32))
        block_type_data.add('driver_type', dtype=np.uint8)
        block_type_data.add('sink_key', self.connections
                            .block_key.values.astype(np.int32))
        block_type_data.add('sink_type', dtype=np.uint8)
        block_type_data.add('delay_type', dtype=np.uint8)

        driver_and_sink_type(block_type_data.v['driver_key'],
                             block_type_data.v['sink_key'],
                             block_type_data.v['driver_type'],
                             block_type_data.v['sink_type'],
                             block_type_data.v['block_type'])
        connection_delay_type(block_type_data.v['driver_type'],
                              block_type_data.v['sink_type'],
                              block_type_data.v['delay_type'])

        for c in ('driver_type', 'sink_type', 'delay_type'):
            self.connections[c] = block_type_data[c].values
        self.connections.drop('block_type', axis=1, inplace=True)
        self.connections.rename(columns={'block_key': 'sink_key',
                                         'driver_block_key': 'driver_key'},
                                inplace=True)

    def driver_connections(self):
        # All driver connections
        # ======================
        #
        # Include:
        #
        #  - All input block connections
        return self.connections.loc[self.connections.type
                                    .isin([CONNECTION_DRIVER,
                                           CONNECTION_CLOCK_DRIVER])]

    def sink_connections(self):
        # All driver connections
        # ======================
        #
        # Include:
        #
        #  - All input block connections
        return self.connections.loc[self.connections.type == CONNECTION_SINK]

    def clock_connections(self):
        # All synchronous/clock connections
        # =================================
        #
        # Include:
        #
        #  - All connections where the driver is a logic block.
        query = ((self.connections.type == CONNECTION_DRIVER) &
                 self.connections.synchronous & (self.connections.sink_type ==
                                                 LOGIC_BLOCK))
        return self.connections.loc[query]

    def input_block_keys(self):
        keys = (self.connections[self.connections.sink_type ==
                                 INPUT_BLOCK].sink_key.unique())
        keys.sort()
        return keys

    def output_block_keys(self):
        keys = (self.connections[self.connections.sink_type ==
                                 OUTPUT_BLOCK].sink_key.unique())
        keys.sort()
        return keys

    def io_block_keys(self):
        keys = (self.connections[self.connections.sink_type
                                 .isin(set([INPUT_BLOCK,
                                            OUTPUT_BLOCK]))].sink_key
                .unique())
        keys.sort()
        return keys

    def sync_logic_block_keys(self):
        keys = self.clock_connections().sink_key.unique()
        keys.sort()
        return keys

    def logic_block_keys(self):
        keys = self.connections[self.connections.sink_type ==
                                LOGIC_BLOCK].sink_key.unique()
        keys.sort()
        return keys

    def __len__(self):
        return len(self.connections)


def get_simple_net_list_frame():
    '''
    Return a `ConnectionsTable` table instance for the net-list below:

           ┌───┐
           │ 0 │─┐
           └───┘ │
                 │   ┌───┐
        ┌────────┴───│ 3 │──┐
        │        ┌───└───┘  │
        │  ┌───┐ │          └──┌───┐
        │  │ 1 │─┼─────────────│ 5 │──┐
        │  └───┘ │             └───┘  │
        │        └────────────────┐   │  ╔═══╗
        └──────────┐              │   └──║ 6 ║──┐
                   │              └──────╚═══╝  │
           ┌───┐   └─╔═══╗                      └──┌───┐     ┌───┐
           │ 2 │─────║ 4 ║─────────────────────────│ 7 │─────│ 8 │
           └───┘     ╚═══╝                         └───┘     └───┘

    Note that blocks 4 and 6 are synchronous, _i.e., they are connected to a
    clock signal and act as signal path end-points.
    '''
    connections = pd.DataFrame(np.array([[0, 0, 0], [1, 1, 0], [2, 2, 0],
                                         [3, 0, 0], [3, 1, 1], [3, 3, 4],
                                         [4, 0, 0], [4, 2, 1], [4, 4, 4],
                                         [4, 8, 5], [5, 1, 0], [5, 3, 1],
                                         [5, 5, 4], [6, 8, 5], [6, 1, 0],
                                         [6, 5, 1], [6, 6, 4], [7, 4, 0],
                                         [7, 6, 1], [7, 7, 4], [8, 7, 0],
                                         [9, 8, 0]]),
                               columns=['block_key', 'net_key', 'pin_key'])
    connections['block_type'] = LOGIC_BLOCK
    connections['block_type'][:3] = INPUT_BLOCK
    connections['block_type'][-2:-1] = OUTPUT_BLOCK
    connections['block_type'][-1:] = INPUT_BLOCK
    connections['type'] = CONNECTION_SINK

    connections.loc[(connections.pin_key == LOGIC_DRIVER_PIN) |
                    ((connections.pin_key == INPUT_DRIVER_PIN) &
                    (connections.block_type == INPUT_BLOCK)), 'type'] = \
        CONNECTION_DRIVER
    connections.loc[((connections.pin_key == CLOCK_PIN) &
                    (connections.block_type == LOGIC_BLOCK)), 'type'] = \
        CONNECTION_CLOCK
    connections.type.iloc[-1] = CONNECTION_CLOCK_DRIVER
    connections = connections.sort('net_key')
    return connections


def get_simple_net_list():
    # Manually constructed arrival times and departure times for testing.
    arrival_times = np.array([0, 0, 0, 1, 1, 2, 3, 1, 2], dtype=np.float32)
    departure_times = np.array([3, 3, 1, 2, 2, 1, 2, 1, 0], dtype=np.float32)

    return (ConnectionsTable(get_simple_net_list_frame()), arrival_times,
            departure_times)


def parse_args(argv=None):
    '''Parses arguments, returns (options, args).'''
    from argparse import ArgumentParser

    if argv is None:
        argv = sys.argv

    parser = ArgumentParser(description='Prepare net-list connection table.')
    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument('-g', '--include-globals', action='store_true')
    mutex_group.add_argument('-l', '--include-labels', action='store_true')
    parser.add_argument(dest='net_file_namebase')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    connections, synchronous_blocks, block_count, net_count = \
        get_connections_table(args.net_file_namebase,
                              labels=args.include_labels)
