"""Fixed raw-log feature decoder; input-only dependency closure."""
import json
import numpy as np
from scipy import sparse
ENUMS = {'action': ['allow', 'deny', 'open', 'close'], 'outcome': ['success', 'failure', 'allowed', 'blocked'], 'credential_check': ['valid', 'invalid'], 'response': ['missing'], 'src_role': ['inside', 'outside', 'dmz', 'identity', 'internet'], 'dst_role': ['inside', 'outside', 'dmz', 'identity', 'internet'], 'transport_protocol': ['tcp', 'udp', 'icmp', 'icmp6'] + ['ipproto_' + str(i) for i in range(256)], 'application_protocol': ['http', 'https', 'http1', 'http1.0', 'http1.1', 'http2', 'http3', 'tls', 'tlsv1', 'tlsv1.0', 'tlsv1.1', 'tlsv1.2', 'tlsv1.3', 'tls1.0', 'tls1.1', 'tls1.2', 'tls1.3'], 'src_port_range': ['system', 'user', 'dynamic'], 'dst_port_range': ['system', 'user', 'dynamic'], 'operation_record': ['path', 'syscall', 'cwd', 'proctitle', 'execve'], 'authentication_protocol': ['kerberos', 'ntlm', 'negotiate', 'msv1_0', 'digest']}
NUMERIC = ('attempt_count', 'duration_seconds', 'bytes', 'bytes_in', 'bytes_out', 'packets', 'keylength')
BIT_FIELDS = {'icmp_type': (9, 256), 'icmp_code': (9, 256), 'http_status': (10, 1023), 'logontype': (9, 256), 'status': (33, 2 ** 32), 'substatus': (33, 2 ** 32), 'src_port_fixed': (17, 65536), 'dst_port_fixed': (17, 65536)}

class FixedFacts:

    def __init__(self, view):
        self.view = view
        self.feature_names = []
        for k, values in ENUMS.items():
            self.feature_names.extend((k + '=' + v for v in values))
        self.feature_names.extend(('numeric:' + k for k in NUMERIC))
        for k, (width, missing) in BIT_FIELDS.items():
            if k == 'src_port_fixed' and view != 'C_BOTH':
                continue
            if k == 'dst_port_fixed' and view not in ('B_DESTINATION', 'C_BOTH'):
                continue
            self.feature_names.extend((k + ':bit' + str(i) for i in range(width)))
        self.feature_names.extend(['exit_signed', 'syscall_ordinal'])
        self.index = {name: i for i, name in enumerate(self.feature_names)}
        self.scale = np.ones(len(self.feature_names))

    def raw(self, values):
        rows = []
        cols = []
        data = []
        self.last_unknown = {}

        def add(r, name, value):
            if value:
                rows.append(r)
                cols.append(self.index[name])
                data.append(float(value))
        for row, f in enumerate(values):
            f = json.loads(f) if isinstance(f, str) else f
            allowed = set(ENUMS) | set(NUMERIC) | set(BIT_FIELDS) | {'exit_category', 'syscall_category'}
            for key in set(f) - allowed:
                self.last_unknown[key] = self.last_unknown.get(key, 0) + 1
            if set(f) & {'category', 'src_port_category', 'dst_port_category', 'product', 'timestamp'}:
                raise ValueError('Forbidden shortcut or lexical port bypass')
            for k, valueset in ENUMS.items():
                if k in f:
                    value = str(f[k]).casefold()
                    if value in valueset:
                        add(row, k + '=' + value, 1)
                    else:
                        self.last_unknown[k + '=' + value] = self.last_unknown.get(k + '=' + value, 0) + 1
            for k in NUMERIC:
                if k in f:
                    n = float(f[k])
                    if not np.isfinite(n) or n < 0:
                        raise ValueError('Invalid numeric observation')
                    add(row, 'numeric:' + k, 1 + np.log1p(n))
            for k, (width, missing) in BIT_FIELDS.items():
                if k + ':bit0' not in self.index:
                    continue
                n = f.get(k, missing)
                if k in ('status', 'substatus') and isinstance(n, str):
                    n = int(n, 16)
                if isinstance(n, bool) or int(n) != n or (not 0 <= int(n) <= missing):
                    raise ValueError('Invalid finite fact')
                for bit in range(width):
                    add(row, k + ':bit' + str(bit), int(n) >> bit & 1)
            if 'exit_category' in f:
                n = int(f['exit_category'])
                add(row, 'exit_signed', np.sign(n) * np.log1p(abs(n)))
            if 'syscall_category' in f:
                self.last_unknown['audit_only:syscall_category'] = self.last_unknown.get('audit_only:syscall_category', 0) + 1
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(values), len(self.feature_names)), dtype=np.float64)

    def fit(self, values):
        x = self.raw(values)
        maxabs = np.asarray(abs(x).max(axis=0).toarray()).ravel() if len(values) else np.zeros(x.shape[1])
        for i, n in enumerate(self.feature_names):
            if n.startswith('numeric:') or n == 'exit_signed':
                self.scale[i] = max(1, maxabs[i])
        self.fit_unknown = dict(self.last_unknown)
        return self

    def transform(self, values):
        return self.raw(values).multiply(1 / self.scale).tocsr()

    def names(self):
        return np.asarray(self.feature_names)
