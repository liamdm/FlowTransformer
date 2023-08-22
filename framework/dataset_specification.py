#  FlowTransformer 2023 by liamdm / liam@riftcs.com
from typing import List, Optional


class DatasetSpecification:
    def  __init__(self, include_fields:List[str], categorical_fields:List[str], class_column:str, benign_label:str, test_column:Optional[str]=None):
        """
        Defines the format of specific NIDS dataset
        :param include_fields: The fields to include as part of classification
        :param categorical_fields: Fields that should be treated as categorical
        :param class_column: The column name that includes the class of the flow, eg. DDoS or Benign
        :param benign_label: The label of benign traffic, eg. Benign or 0
        :param test_column: The column indicating if this row is a member of the test or training dataset
        """
        self.include_fields:List[str] = include_fields
        self.categorical_fields:List[str] = categorical_fields
        self.class_column = class_column
        self.benign_label = benign_label
        self.test_column:Optional[str] = test_column

class NamedDatasetSpecifications:
    """
    Example specifications of some common datasets
    """
    
    cse_cic_ids_2018 = DatasetSpecification(
            include_fields=['NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES', 'OUT_PKTS', 'OUT_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN', 'L4_SRC_PORT', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'IN_PKTS', 'NUM_PKTS_512_TO_1024_BYTES', 'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN', 'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'LONGEST_FLOW_PKT', 'L4_DST_PORT', 'MIN_TTL', 'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS', 'TCP_FLAGS', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE', 'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES', 'IN_BYTES', 'RETRANSMITTED_IN_BYTES', 'TCP_WIN_MAX_OUT', 'L7_PROTO', 'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS'],
            categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', 'TCP_FLAGS', 'ICMP_IPV4_TYPE', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'L4_DST_PORT', 'L7_PROTO'],
            class_column="Attack",
            benign_label="Benign"
        )

    unified_flow_format = DatasetSpecification(
            include_fields=['NUM_PKTS_UP_TO_128_BYTES', 'SRC_TO_DST_SECOND_BYTES', 'OUT_PKTS', 'OUT_BYTES', 'NUM_PKTS_128_TO_256_BYTES', 'DST_TO_SRC_AVG_THROUGHPUT', 'DURATION_IN', 'L4_SRC_PORT', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'IN_PKTS', 'NUM_PKTS_512_TO_1024_BYTES', 'CLIENT_TCP_FLAGS', 'TCP_WIN_MAX_IN', 'NUM_PKTS_256_TO_512_BYTES', 'SHORTEST_FLOW_PKT', 'MIN_IP_PKT_LEN', 'LONGEST_FLOW_PKT', 'L4_DST_PORT', 'MIN_TTL', 'DST_TO_SRC_SECOND_BYTES', 'NUM_PKTS_1024_TO_1514_BYTES', 'DURATION_OUT', 'FLOW_DURATION_MILLISECONDS', 'TCP_FLAGS', 'MAX_TTL', 'SRC_TO_DST_AVG_THROUGHPUT', 'ICMP_IPV4_TYPE', 'MAX_IP_PKT_LEN', 'RETRANSMITTED_OUT_BYTES', 'IN_BYTES', 'RETRANSMITTED_IN_BYTES', 'TCP_WIN_MAX_OUT', 'L7_PROTO', 'RETRANSMITTED_OUT_PKTS', 'RETRANSMITTED_IN_PKTS'],
            categorical_fields=['CLIENT_TCP_FLAGS', 'L4_SRC_PORT', 'TCP_FLAGS', 'ICMP_IPV4_TYPE', 'ICMP_TYPE', 'PROTOCOL', 'SERVER_TCP_FLAGS', 'L4_DST_PORT', 'L7_PROTO'],
            class_column="Attack",
            benign_label="Benign"
        )

    mqtt = DatasetSpecification(
        include_fields=['prt_src', 'prt_dst', 'proto', 'fwd_num_pkts', 'bwd_num_pkts', 'fwd_mean_iat', 'bwd_mean_iat',
                   'fwd_std_iat', 'bwd_std_iat', 'fwd_min_iat', 'bwd_min_iat', 'fwd_max_iat', 'bwd_max_iat',
                   'fwd_mean_pkt_len', 'bwd_mean_pkt_len', 'fwd_std_pkt_len', 'bwd_std_pkt_len', 'fwd_min_pkt_len',
                   'bwd_min_pkt_len', 'fwd_max_pkt_len', 'bwd_max_pkt_len', 'fwd_num_bytes', 'bwd_num_bytes',
                   'fwd_num_psh_flags', 'bwd_num_psh_flags', 'fwd_num_rst_flags', 'bwd_num_rst_flags',
                   'fwd_num_urg_flags', 'bwd_num_urg_flags'],
        categorical_fields=['prt_src', 'prt_dst', 'proto'],
        class_column='is_attack',
        benign_label='1',
    )

    nsl_kdd = DatasetSpecification(
        include_fields=['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
                   'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'],
        categorical_fields=['protocol_type', 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login'],
        benign_label='normal',
        class_column='class',
        test_column='is_test'
    )