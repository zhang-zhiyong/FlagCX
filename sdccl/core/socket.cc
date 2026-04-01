/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE-NCCL.txt for license information
 ************************************************************************/

#include "socket.h"
#include "check.h"
#include "debug.h"
#include "param.h"
#include "utils.h"
#include <cstddef>
#include <stdlib.h>

#include <cerrno>
#include <ifaddrs.h>
#include <net/if.h>
#include <string.h>
#include <unistd.h>

static sdcclResult_t socketProgressOpt(int op, struct sdcclSocket *sock,
                                        void *ptr, int size, int *offset,
                                        int block, int *closed) {
  int bytes = 0;
  *closed = 0;
  char *data = (char *)ptr;
  char line[SOCKET_NAME_MAXLEN + 1];
  do {
    if (op == SDCCL_SOCKET_RECV)
      bytes = recv(sock->fd, data + (*offset), size - (*offset),
                   block ? 0 : MSG_DONTWAIT);
    if (op == SDCCL_SOCKET_SEND)
      bytes = send(sock->fd, data + (*offset), size - (*offset),
                   block ? MSG_NOSIGNAL : MSG_DONTWAIT | MSG_NOSIGNAL);
    if (op == SDCCL_SOCKET_RECV && bytes == 0) {
      *closed = 1;
      return sdcclSuccess;
    }
    if (bytes == -1) {
      if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
        WARN("socketProgressOpt: Call to recv from %s failed : %s",
             sdcclSocketToString(&sock->addr, line), strerror(errno));
        return sdcclRemoteError;
      } else {
        bytes = 0;
      }
    }
    (*offset) += bytes;
    if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED)) {
      INFO(SDCCL_NET, "socketProgressOpt: abort called");
      return sdcclInternalError;
    }
  } while (bytes > 0 && (*offset) < size);
  return sdcclSuccess;
}

static sdcclResult_t socketProgress(int op, struct sdcclSocket *sock,
                                     void *ptr, int size, int *offset) {
  int closed;
  SDCCLCHECK(
      socketProgressOpt(op, sock, ptr, size, offset, 0 /*block*/, &closed));
  if (closed) {
    char line[SOCKET_NAME_MAXLEN + 1];
    WARN("socketProgress: Connection closed by remote peer %s",
         sdcclSocketToString(&sock->addr, line, 0));
    return sdcclRemoteError;
  }
  return sdcclSuccess;
}

static sdcclResult_t socketWait(int op, struct sdcclSocket *sock, void *ptr,
                                 int size, int *offset) {
  while (*offset < size)
    SDCCLCHECK(socketProgress(op, sock, ptr, size, offset));
  return sdcclSuccess;
}

/* Format a string representation of a (union sdcclSocketAddress *) socket
 * address using getnameinfo()
 *
 * Output: "IPv4/IPv6 address<port>"
 */
const char *sdcclSocketToString(union sdcclSocketAddress *addr, char *buf,
                                 const int numericHostForm /*= 1*/) {
  if (buf == NULL || addr == NULL)
    return NULL;
  struct sockaddr *saddr = &addr->sa;
  if (saddr->sa_family != AF_INET && saddr->sa_family != AF_INET6) {
    buf[0] = '\0';
    return buf;
  }
  char host[NI_MAXHOST], service[NI_MAXSERV];
  /* NI_NUMERICHOST: If set, then the numeric form of the hostname is returned.
   * (When not set, this will still happen in case the node's name cannot be
   * determined.)
   */
  int flag = NI_NUMERICSERV | (numericHostForm ? NI_NUMERICHOST : 0);
  (void)getnameinfo(saddr, sizeof(union sdcclSocketAddress), host, NI_MAXHOST,
                    service, NI_MAXSERV, flag);
  sprintf(buf, "%s<%s>", host, service);
  return buf;
}

static uint16_t socketToPort(union sdcclSocketAddress *addr) {
  struct sockaddr *saddr = &addr->sa;
  return ntohs(saddr->sa_family == AF_INET ? addr->sin.sin_port
                                           : addr->sin6.sin6_port);
}

/* Allow the user to force the IPv4/IPv6 interface selection */
static int envSocketFamily(void) {
  int family = -1; // Family selection is not forced, will use first one found
  const char *env = sdcclGetEnv("SDCCL_SOCKET_FAMILY");
  if (env == NULL)
    return family;

  INFO(SDCCL_ENV, "SDCCL_SOCKET_FAMILY set by environment to %s", env);

  if (strcmp(env, "AF_INET") == 0)
    family = AF_INET; // IPv4
  else if (strcmp(env, "AF_INET6") == 0)
    family = AF_INET6; // IPv6
  return family;
}

static int findInterfaces(const char *prefixList, char *names,
                          union sdcclSocketAddress *addrs, int sock_family,
                          int maxIfNameSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  struct netIf userIfs[MAX_IFS];
  bool searchNot = prefixList && prefixList[0] == '^';
  if (searchNot)
    prefixList++;
  bool searchExact = prefixList && prefixList[0] == '=';
  if (searchExact)
    prefixList++;
  int nUserIfs = parseStringList(prefixList, userIfs, MAX_IFS);

  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && found < maxIfs;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL)
      continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    TRACE(SDCCL_INIT | SDCCL_NET, "Found interface %s:%s",
          interface->ifa_name,
          sdcclSocketToString((union sdcclSocketAddress *)interface->ifa_addr,
                               line));

    /* Allow the caller to force the socket family type */
    if (sock_family != -1 && family != sock_family)
      continue;

    /* We also need to skip IPv6 loopback interfaces */
    if (family == AF_INET6) {
      struct sockaddr_in6 *sa = (struct sockaddr_in6 *)(interface->ifa_addr);
      if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr))
        continue;
    }

    // check against user specified interfaces
    if (!(matchIfList(interface->ifa_name, -1, userIfs, nUserIfs, searchExact) ^
          searchNot)) {
      continue;
    }

    // Check that this interface has not already been saved
    // getifaddrs() normal order appears to be; IPv4, IPv6 Global, IPv6 Link
    bool duplicate = false;
    for (int i = 0; i < found; i++) {
      if (strcmp(interface->ifa_name, names + i * maxIfNameSize) == 0) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      // Store the interface name
      strncpy(names + found * maxIfNameSize, interface->ifa_name,
              maxIfNameSize);
      // Store the IP address
      int salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                      : sizeof(struct sockaddr_in6);
      memcpy(addrs + found, interface->ifa_addr, salen);
      found++;
    }
  }

  freeifaddrs(interfaces);
  return found;
}

static bool matchSubnet(struct ifaddrs local_if,
                        union sdcclSocketAddress *remote) {
  /* Check family first */
  int family = local_if.ifa_addr->sa_family;
  if (family != remote->sa.sa_family) {
    return false;
  }

  if (family == AF_INET) {
    struct sockaddr_in *local_addr = (struct sockaddr_in *)(local_if.ifa_addr);
    struct sockaddr_in *mask = (struct sockaddr_in *)(local_if.ifa_netmask);
    struct sockaddr_in &remote_addr = remote->sin;
    struct in_addr local_subnet, remote_subnet;
    local_subnet.s_addr = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
    remote_subnet.s_addr = remote_addr.sin_addr.s_addr & mask->sin_addr.s_addr;
    return (local_subnet.s_addr ^ remote_subnet.s_addr) ? false : true;
  } else if (family == AF_INET6) {
    struct sockaddr_in6 *local_addr =
        (struct sockaddr_in6 *)(local_if.ifa_addr);
    struct sockaddr_in6 *mask = (struct sockaddr_in6 *)(local_if.ifa_netmask);
    struct sockaddr_in6 &remote_addr = remote->sin6;
    struct in6_addr &local_in6 = local_addr->sin6_addr;
    struct in6_addr &mask_in6 = mask->sin6_addr;
    struct in6_addr &remote_in6 = remote_addr.sin6_addr;
    bool same = true;
    int len = 16;                   // IPv6 address is 16 unsigned char
    for (int c = 0; c < len; c++) { // Network byte order is big-endian
      char c1 = local_in6.s6_addr[c] & mask_in6.s6_addr[c];
      char c2 = remote_in6.s6_addr[c] & mask_in6.s6_addr[c];
      if (c1 ^ c2) {
        same = false;
        break;
      }
    }
    // At last, we need to compare scope id
    // Two Link-type addresses can have the same subnet address even though they
    // are not in the same scope For Global type, this field is 0, so a
    // comparison wouldn't matter
    same &= (local_addr->sin6_scope_id == remote_addr.sin6_scope_id);
    return same;
  } else {
    WARN("Net : Unsupported address family type");
    return false;
  }
}

int sdcclFindInterfaceMatchSubnet(char *ifNames,
                                   union sdcclSocketAddress *localAddrs,
                                   union sdcclSocketAddress *remoteAddr,
                                   int ifNameMaxSize, int maxIfs) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  char line_a[SOCKET_NAME_MAXLEN + 1];
  int found = 0;
  struct ifaddrs *interfaces, *interface;
  getifaddrs(&interfaces);
  for (interface = interfaces; interface && !found;
       interface = interface->ifa_next) {
    if (interface->ifa_addr == NULL)
      continue;

    /* We only support IPv4 & IPv6 */
    int family = interface->ifa_addr->sa_family;
    if (family != AF_INET && family != AF_INET6)
      continue;

    // check against user specified interfaces
    if (!matchSubnet(*interface, remoteAddr)) {
      continue;
    }

    // Store the local IP address
    int salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                    : sizeof(struct sockaddr_in6);
    memcpy(localAddrs + found, interface->ifa_addr, salen);

    // Store the interface name
    strncpy(ifNames + found * ifNameMaxSize, interface->ifa_name,
            ifNameMaxSize);

    TRACE(SDCCL_INIT | SDCCL_NET,
          "NET : Found interface %s:%s in the same subnet as remote address %s",
          interface->ifa_name, sdcclSocketToString(localAddrs + found, line),
          sdcclSocketToString(remoteAddr, line_a));
    found++;
    if (found == maxIfs)
      break;
  }

  if (found == 0) {
    WARN("Net : No interface found in the same subnet as remote address %s",
         sdcclSocketToString(remoteAddr, line_a));
  }
  freeifaddrs(interfaces);
  return found;
}

sdcclResult_t sdcclSocketGetAddrFromString(union sdcclSocketAddress *ua,
                                             const char *ip_port_pair) {
  if (!(ip_port_pair && strlen(ip_port_pair) > 1)) {
    WARN("Net : string is null");
    return sdcclInvalidArgument;
  }

  bool ipv6 = ip_port_pair[0] == '[';
  /* Construct the sockaddress structure */
  if (!ipv6) {
    struct netIf ni;
    // parse <ip_or_hostname>:<port> string, expect one pair
    if (parseStringList(ip_port_pair, &ni, 1) != 1) {
      WARN("Net : No valid <IPv4_or_hostname>:<port> pair found");
      return sdcclInvalidArgument;
    }

    struct addrinfo hints, *p;
    int rv;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;

    if ((rv = getaddrinfo(ni.prefix, NULL, &hints, &p)) != 0) {
      WARN("Net : error encountered when getting address info : %s",
           gai_strerror(rv));
      return sdcclInvalidArgument;
    }

    // use the first
    if (p->ai_family == AF_INET) {
      struct sockaddr_in &sin = ua->sin;
      memcpy(&sin, p->ai_addr, sizeof(struct sockaddr_in));
      sin.sin_family = AF_INET; // IPv4
      // inet_pton(AF_INET, ni.prefix, &(sin.sin_addr));  // IP address
      sin.sin_port = htons(ni.port); // port
    } else if (p->ai_family == AF_INET6) {
      struct sockaddr_in6 &sin6 = ua->sin6;
      memcpy(&sin6, p->ai_addr, sizeof(struct sockaddr_in6));
      sin6.sin6_family = AF_INET6;     // IPv6
      sin6.sin6_port = htons(ni.port); // port
      sin6.sin6_flowinfo = 0;          // needed by IPv6, but possibly obsolete
      sin6.sin6_scope_id = 0;          // should be global scope, set to 0
    } else {
      WARN("Net : unsupported IP family");
      return sdcclInvalidArgument;
    }

    freeaddrinfo(p); // all done with this structure

  } else {
    int i, j = -1, len = strlen(ip_port_pair);
    for (i = 1; i < len; i++) {
      if (ip_port_pair[i] == '%')
        j = i;
      if (ip_port_pair[i] == ']')
        break;
    }
    if (i == len) {
      WARN("Net : No valid [IPv6]:port pair found");
      return sdcclInvalidArgument;
    }
    bool global_scope =
        (j == -1 ? true
                 : false); // If no % found, global scope; otherwise, link scope

    char ip_str[NI_MAXHOST], port_str[NI_MAXSERV], if_name[IFNAMSIZ];
    memset(ip_str, '\0', sizeof(ip_str));
    memset(port_str, '\0', sizeof(port_str));
    memset(if_name, '\0', sizeof(if_name));
    strncpy(ip_str, ip_port_pair + 1, global_scope ? i - 1 : j - 1);
    strncpy(port_str, ip_port_pair + i + 2, len - i - 1);
    int port = atoi(port_str);
    if (!global_scope)
      strncpy(if_name, ip_port_pair + j + 1,
              i - j - 1); // If not global scope, we need the intf name

    struct sockaddr_in6 &sin6 = ua->sin6;
    sin6.sin6_family = AF_INET6;                    // IPv6
    inet_pton(AF_INET6, ip_str, &(sin6.sin6_addr)); // IP address
    sin6.sin6_port = htons(port);                   // port
    sin6.sin6_flowinfo = 0; // needed by IPv6, but possibly obsolete
    sin6.sin6_scope_id =
        global_scope
            ? 0
            : if_nametoindex(
                  if_name); // 0 if global scope; intf index if link scope
  }
  return sdcclSuccess;
}

int sdcclFindInterfaces(char *ifNames, union sdcclSocketAddress *ifAddrs,
                         int ifNameMaxSize, int maxIfs) {
  static int shownIfName = 0;
  int nIfs = 0;
  // Allow user to force the INET socket family selection
  int sock_family = envSocketFamily();
  // User specified interface
  const char *env = sdcclGetEnv("SDCCL_SOCKET_IFNAME");
  if (env && strlen(env) > 1) {
    INFO(SDCCL_ENV, "SDCCL_SOCKET_IFNAME set by environment to %s", env);
    // Specified by user : find or fail
    if (shownIfName++ == 0)
      INFO(SDCCL_NET, "SDCCL_SOCKET_IFNAME set to %s", env);
    nIfs = findInterfaces(env, ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
  } else {
    // Try to automatically pick the right one
    // Start with IB
    nIfs = findInterfaces("ib", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                          maxIfs);
    // else see if we can get some hint from COMM ID
    if (nIfs == 0) {
      const char *commId = sdcclGetEnv("SDCCL_COMM_ID");
      if (commId && strlen(commId) > 1) {
        INFO(SDCCL_ENV, "SDCCL_COMM_ID set by environment to %s", commId);
        // Try to find interface that is in the same subnet as the IP in comm id
        union sdcclSocketAddress idAddr;
        sdcclSocketGetAddrFromString(&idAddr, commId);
        nIfs = sdcclFindInterfaceMatchSubnet(ifNames, ifAddrs, &idAddr,
                                              ifNameMaxSize, maxIfs);
      }
    }
    // Then look for anything else (but not docker or lo)
    if (nIfs == 0)
      nIfs = findInterfaces("^docker,lo", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    // Finally look for docker, then lo.
    if (nIfs == 0)
      nIfs = findInterfaces("docker", ifNames, ifAddrs, sock_family,
                            ifNameMaxSize, maxIfs);
    if (nIfs == 0)
      nIfs = findInterfaces("lo", ifNames, ifAddrs, sock_family, ifNameMaxSize,
                            maxIfs);
  }
  return nIfs;
}

sdcclResult_t sdcclSocketListen(struct sdcclSocket *sock) {
  if (sock == NULL) {
    WARN("sdcclSocketListen: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("sdcclSocketListen: file descriptor is -1");
    return sdcclInvalidArgument;
  }

  if (socketToPort(&sock->addr)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt,
                        sizeof(opt)),
             "setsockopt");
#else
    SYSCHECK(setsockopt(sock->fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)),
             "setsockopt");
#endif
  }

  // addr port should be 0 (Any port)
  SYSCHECK(bind(sock->fd, &sock->addr.sa, sock->salen), "bind");

  /* Get the assigned Port */
  socklen_t size = sock->salen;
  SYSCHECK(getsockname(sock->fd, &sock->addr.sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
  TRACE(SDCCL_INIT | SDCCL_NET, "Listening on socket %s",
        sdcclSocketToString(&sock->addr, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in
   * /proc/sys/net/core/somaxconn
   */
  SYSCHECK(listen(sock->fd, 16384), "listen");
  sock->state = sdcclSocketStateReady;
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketGetAddr(struct sdcclSocket *sock,
                                   union sdcclSocketAddress *addr) {
  if (sock == NULL) {
    WARN("sdcclSocketGetAddr: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (sock->state != sdcclSocketStateReady)
    return sdcclInternalError;
  memcpy(addr, &sock->addr, sizeof(union sdcclSocketAddress));
  return sdcclSuccess;
}

static sdcclResult_t socketTryAccept(struct sdcclSocket *sock) {
  socklen_t socklen = sizeof(union sdcclSocketAddress);
  sock->fd = accept(sock->acceptFd, &sock->addr.sa, &socklen);
  if (sock->fd != -1) {
    sock->state = sdcclSocketStateAccepted;
  } else if (errno != EAGAIN && errno != EWOULDBLOCK) {
    WARN("socketTryAccept: Accept failed: %s", strerror(errno));
    return sdcclSystemError;
  }
  return sdcclSuccess;
}

static sdcclResult_t socketFinalizeAccept(struct sdcclSocket *sock) {
  uint64_t magic;
  enum sdcclSocketType type;
  int received = 0;
  const int one = 1;
  SYSCHECK(
      setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(int)),
      "setsockopt");

  SDCCLCHECK(sdcclSocketProgress(SDCCL_SOCKET_RECV, sock, &magic,
                                   sizeof(magic), &received));
  if (received == 0)
    return sdcclSuccess;
  SDCCLCHECK(
      socketWait(SDCCL_SOCKET_RECV, sock, &magic, sizeof(magic), &received));
  if (magic != sock->magic) {
    WARN("socketFinalizeAccept: wrong magic %lx != %lx", magic, sock->magic);
    close(sock->fd);
    sock->fd = -1;
    // Ignore spurious connection and accept again
    sock->state = sdcclSocketStateAccepting;
    return sdcclSuccess;
  } else {
    received = 0;
    SDCCLCHECK(
        socketWait(SDCCL_SOCKET_RECV, sock, &type, sizeof(type), &received));
    if (type != sock->type) {
      WARN("socketFinalizeAccept: wrong type %d != %d", type, sock->type);
      sock->state = sdcclSocketStateError;
      close(sock->fd);
      sock->fd = -1;
      return sdcclInternalError;
    } else {
      sock->state = sdcclSocketStateReady;
    }
  }
  return sdcclSuccess;
}

static sdcclResult_t socketStartConnect(struct sdcclSocket *sock) {
  /* blocking/non-blocking connect() is determined by asyncFlag. */
  int ret = connect(sock->fd, &sock->addr.sa, sock->salen);

  if (ret == 0) {
    sock->state = sdcclSocketStateConnected;
    return sdcclSuccess;
  } else if (errno == EINPROGRESS) {
    sock->state = sdcclSocketStateConnectPolling;
    return sdcclSuccess;
  } else if (errno == ECONNREFUSED) {
    if (++sock->refusedRetries == RETRY_REFUSED_TIMES) {
      sock->state = sdcclSocketStateError;
      WARN("socketStartConnect: exceeded retries (%d)", sock->refusedRetries);
      return sdcclRemoteError;
    }
    usleep(SLEEP_INT);
    if (sock->refusedRetries % 1000 == 0)
      INFO(SDCCL_ALL, "Call to connect returned %s, retrying",
           strerror(errno));
    return sdcclSuccess;
  } else if (errno == ETIMEDOUT) {
    if (++sock->timedOutRetries == RETRY_TIMEDOUT_TIMES) {
      sock->state = sdcclSocketStateError;
      WARN("socketStartConnect: exceeded timeouts (%d)", sock->timedOutRetries);
      return sdcclRemoteError;
    }
    usleep(SLEEP_INT);
    return sdcclSuccess;
  } else {
    char line[SOCKET_NAME_MAXLEN + 1];
    sock->state = sdcclSocketStateError;
    WARN("socketStartConnect: Connect to %s failed : %s",
         sdcclSocketToString(&sock->addr, line), strerror(errno));
    return sdcclSystemError;
  }
}

static sdcclResult_t socketPollConnect(struct sdcclSocket *sock) {
  struct pollfd pfd;
  int timeout = 1, ret;
  socklen_t rlen = sizeof(int);

  memset(&pfd, 0, sizeof(struct pollfd));
  pfd.fd = sock->fd;
  pfd.events = POLLOUT;
  ret = poll(&pfd, 1, timeout);

  if (ret == 0 || (ret < 0 && errno == EINTR)) {
    return sdcclSuccess;
  } else if (ret < 0) {
    WARN("socketPollConnect poll() failed with error %s", strerror(errno));
    return sdcclRemoteError;
  } else {
    EQCHECK(ret == 1 && (pfd.revents & POLLOUT), 0);
  }

  /* check socket status */
  SYSCHECK(getsockopt(sock->fd, SOL_SOCKET, SO_ERROR, (void *)&ret, &rlen),
           "getsockopt");

  if (ret == 0) {
    sock->state = sdcclSocketStateConnected;
  } else if (ret == ECONNREFUSED) {
    if (++sock->refusedRetries == RETRY_REFUSED_TIMES) {
      sock->state = sdcclSocketStateError;
      WARN("socketPollConnect: exceeded retries (%d)", sock->refusedRetries);
      return sdcclRemoteError;
    }
    if (sock->refusedRetries % 1000 == 0)
      INFO(SDCCL_ALL, "Call to connect returned %s, retrying",
           strerror(errno));
    usleep(SLEEP_INT);
    sock->state = sdcclSocketStateConnecting;
  } else if (ret == ETIMEDOUT) {
    if (++sock->timedOutRetries == RETRY_TIMEDOUT_TIMES) {
      sock->state = sdcclSocketStateError;
      WARN("socketPollConnect: exceeded timeouts (%d)", sock->timedOutRetries);
      return sdcclRemoteError;
    }
    usleep(SLEEP_INT);
    sock->state = sdcclSocketStateConnecting;
  } else if (ret != EINPROGRESS) {
    sock->state = sdcclSocketStateError;
    char line[SOCKET_NAME_MAXLEN + 1];
    WARN("socketPollConnect: Connect to %s returned %d(%s) errno %d(%s)",
         sdcclSocketToString(&sock->addr, line), ret, strerror(ret), errno,
         strerror(errno));
    return sdcclSystemError;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketPollConnect(struct sdcclSocket *sock) {
  if (sock == NULL) {
    WARN("sdcclSocketPollConnect: pass NULL socket");
    return sdcclInvalidArgument;
  }
  SDCCLCHECK(socketPollConnect(sock));
  return sdcclSuccess;
}

static sdcclResult_t socketFinalizeConnect(struct sdcclSocket *sock) {
  int sent = 0;
  SDCCLCHECK(socketProgress(SDCCL_SOCKET_SEND, sock, &sock->magic,
                             sizeof(sock->magic), &sent));
  if (sent == 0)
    return sdcclSuccess;
  SDCCLCHECK(socketWait(SDCCL_SOCKET_SEND, sock, &sock->magic,
                         sizeof(sock->magic), &sent));
  sent = 0;
  SDCCLCHECK(socketWait(SDCCL_SOCKET_SEND, sock, &sock->type,
                         sizeof(sock->type), &sent));
  sock->state = sdcclSocketStateReady;
  return sdcclSuccess;
}

static sdcclResult_t socketProgressState(struct sdcclSocket *sock) {
  if (sock->state == sdcclSocketStateAccepting) {
    SDCCLCHECK(socketTryAccept(sock));
  }
  if (sock->state == sdcclSocketStateAccepted) {
    SDCCLCHECK(socketFinalizeAccept(sock));
  }
  if (sock->state == sdcclSocketStateConnecting) {
    SDCCLCHECK(socketStartConnect(sock));
  }
  if (sock->state == sdcclSocketStateConnectPolling) {
    SDCCLCHECK(socketPollConnect(sock));
  }
  if (sock->state == sdcclSocketStateConnected) {
    SDCCLCHECK(socketFinalizeConnect(sock));
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketReady(struct sdcclSocket *sock, int *running) {
  if (sock == NULL) {
    *running = 0;
    return sdcclSuccess;
  }
  if (sock->state == sdcclSocketStateError ||
      sock->state == sdcclSocketStateClosed) {
    WARN("sdcclSocketReady: unexpected socket state %d", sock->state);
    return sdcclRemoteError;
  }
  *running = (sock->state == sdcclSocketStateReady) ? 1 : 0;
  if (*running == 0) {
    SDCCLCHECK(socketProgressState(sock));
    *running = (sock->state == sdcclSocketStateReady) ? 1 : 0;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketConnect(struct sdcclSocket *sock) {
#ifdef ENABLE_TRACE
  char line[SOCKET_NAME_MAXLEN + 1];
#endif
  const int one = 1;

  if (sock == NULL) {
    WARN("sdcclSocketConnect: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (sock->fd == -1) {
    WARN("sdcclSocketConnect: file descriptor is -1");
    return sdcclInvalidArgument;
  }

  if (sock->state != sdcclSocketStateInitialized) {
    WARN("sdcclSocketConnect: wrong socket state %d", sock->state);
    if (sock->state == sdcclSocketStateError)
      return sdcclRemoteError;
    return sdcclInternalError;
  }
  TRACE(SDCCL_INIT | SDCCL_NET, "Connecting to socket %s",
        sdcclSocketToString(&sock->addr, line));

  SYSCHECK(
      setsockopt(sock->fd, IPPROTO_TCP, TCP_NODELAY, (char *)&one, sizeof(int)),
      "setsockopt");

  sock->state = sdcclSocketStateConnecting;
  do {
    SDCCLCHECK(socketProgressState(sock));
  } while (sock->asyncFlag == 0 &&
           (sock->abortFlag == NULL ||
            __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED) == 0) &&
           (sock->state == sdcclSocketStateConnecting ||
            sock->state == sdcclSocketStateConnectPolling ||
            sock->state == sdcclSocketStateConnected));

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED))
    return sdcclInternalError;

  switch (sock->state) {
    case sdcclSocketStateConnecting:
    case sdcclSocketStateConnectPolling:
    case sdcclSocketStateConnected:
    case sdcclSocketStateReady:
      return sdcclSuccess;
    case sdcclSocketStateError:
      return sdcclSystemError;
    default:
      WARN("sdcclSocketConnect: wrong socket state %d", sock->state);
      return sdcclInternalError;
  }
}

sdcclResult_t sdcclSocketAccept(struct sdcclSocket *sock,
                                  struct sdcclSocket *listenSock) {
  sdcclResult_t ret = sdcclSuccess;

  if (listenSock == NULL || sock == NULL) {
    WARN("sdcclSocketAccept: pass NULL socket");
    ret = sdcclInvalidArgument;
    goto exit;
  }
  if (listenSock->state != sdcclSocketStateReady) {
    WARN("sdcclSocketAccept: wrong socket state %d", listenSock->state);
    if (listenSock->state == sdcclSocketStateError)
      ret = sdcclSystemError;
    else
      ret = sdcclInternalError;
    goto exit;
  }

  if (sock->acceptFd == -1) {
    memcpy(sock, listenSock, sizeof(struct sdcclSocket));
    sock->acceptFd = listenSock->fd;
    sock->state = sdcclSocketStateAccepting;
  }

  do {
    SDCCLCHECKGOTO(socketProgressState(sock), ret, exit);
  } while (sock->asyncFlag == 0 &&
           (sock->abortFlag == NULL ||
            __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED) == 0) &&
           (sock->state == sdcclSocketStateAccepting ||
            sock->state == sdcclSocketStateAccepted));

  if (sock->abortFlag && __atomic_load_n(sock->abortFlag, __ATOMIC_RELAXED))
    return sdcclInternalError;

  switch (sock->state) {
    case sdcclSocketStateAccepting:
    case sdcclSocketStateAccepted:
    case sdcclSocketStateReady:
      ret = sdcclSuccess;
      break;
    case sdcclSocketStateError:
      ret = sdcclSystemError;
      break;
    default:
      WARN("sdcclSocketAccept: wrong socket state %d", sock->state);
      ret = sdcclInternalError;
      break;
  }

exit:
  return ret;
}

sdcclResult_t sdcclSocketInit(struct sdcclSocket *sock,
                                union sdcclSocketAddress *addr, uint64_t magic,
                                enum sdcclSocketType type,
                                volatile uint32_t *abortFlag, int asyncFlag) {
  sdcclResult_t ret = sdcclSuccess;

  if (sock == NULL)
    goto exit;
  sock->timedOutRetries = 0;
  sock->refusedRetries = 0;
  sock->abortFlag = abortFlag;
  sock->asyncFlag = asyncFlag;
  sock->state = sdcclSocketStateInitialized;
  sock->magic = magic;
  sock->type = type;
  sock->fd = -1;
  sock->acceptFd = -1;

  if (addr) {
    /* IPv4/IPv6 support */
    int family;
    memcpy(&sock->addr, addr, sizeof(union sdcclSocketAddress));
    family = sock->addr.sa.sa_family;
    if (family != AF_INET && family != AF_INET6) {
      char line[SOCKET_NAME_MAXLEN + 1];
      WARN("sdcclSocketInit: connecting to address %s with family %d is "
           "neither AF_INET(%d) nor AF_INET6(%d)",
           sdcclSocketToString(&sock->addr, line), family, AF_INET, AF_INET6);
      ret = sdcclInternalError;
      goto fail;
    }
    sock->salen = (family == AF_INET) ? sizeof(struct sockaddr_in)
                                      : sizeof(struct sockaddr_in6);

    /* Connect to a hostname / port */
    sock->fd = socket(family, SOCK_STREAM, 0);
    if (sock->fd == -1) {
      WARN("sdcclSocketInit: Socket creation failed : %s", strerror(errno));
      ret = sdcclSystemError;
      goto fail;
    }
  } else {
    memset(&sock->addr, 0, sizeof(union sdcclSocketAddress));
  }

  /* Set socket as non-blocking if async or if we need to be able to abort */
  if ((sock->asyncFlag || sock->abortFlag) && sock->fd >= 0) {
    int flags;
    EQCHECKGOTO(flags = fcntl(sock->fd, F_GETFL), -1, ret, fail);
    SYSCHECKGOTO(fcntl(sock->fd, F_SETFL, flags | O_NONBLOCK), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

sdcclResult_t sdcclSocketProgress(int op, struct sdcclSocket *sock,
                                    void *ptr, int size, int *offset) {
  if (sock == NULL) {
    WARN("sdcclSocketProgress: pass NULL socket");
    return sdcclInvalidArgument;
  }
  SDCCLCHECK(socketProgress(op, sock, ptr, size, offset));
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketWait(int op, struct sdcclSocket *sock, void *ptr,
                                int size, int *offset) {
  if (sock == NULL) {
    WARN("sdcclSocketWait: pass NULL socket");
    return sdcclInvalidArgument;
  }
  SDCCLCHECK(socketWait(op, sock, ptr, size, offset));
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketSend(struct sdcclSocket *sock, void *ptr,
                                int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("sdcclSocketSend: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (sock->state != sdcclSocketStateReady) {
    WARN("sdcclSocketSend: socket state (%d) is not ready", sock->state);
    return sdcclInternalError;
  }
  SDCCLCHECK(socketWait(SDCCL_SOCKET_SEND, sock, ptr, size, &offset));
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketRecv(struct sdcclSocket *sock, void *ptr,
                                int size) {
  int offset = 0;
  if (sock == NULL) {
    WARN("sdcclSocketRecv: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (sock->state != sdcclSocketStateReady) {
    WARN("sdcclSocketRecv: socket state (%d) is not ready", sock->state);
    return sdcclInternalError;
  }
  SDCCLCHECK(socketWait(SDCCL_SOCKET_RECV, sock, ptr, size, &offset));
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketSendRecv(struct sdcclSocket *sendSock,
                                    void *sendPtr, int sendSize,
                                    struct sdcclSocket *recvSock,
                                    void *recvPtr, int recvSize) {
  int sendOffset = 0, recvOffset = 0;
  if (sendSock == NULL || recvSock == NULL) {
    WARN("sdcclSocketSendRecv: invalid socket %p/%p", sendSock, recvSock);
    return sdcclInternalError;
  }
  if (sendSock->state != sdcclSocketStateReady ||
      recvSock->state != sdcclSocketStateReady) {
    WARN("sdcclSocketSendRecv: socket state (%d/%d) is not ready",
         sendSock->state, recvSock->state);
    return sdcclInternalError;
  }
  while (sendOffset < sendSize || recvOffset < recvSize) {
    if (sendOffset < sendSize)
      SDCCLCHECK(socketProgress(SDCCL_SOCKET_SEND, sendSock, sendPtr,
                                 sendSize, &sendOffset));
    if (recvOffset < recvSize)
      SDCCLCHECK(socketProgress(SDCCL_SOCKET_RECV, recvSock, recvPtr,
                                 recvSize, &recvOffset));
  }
  return sdcclSuccess;
}

// Receive or detect connection closed
sdcclResult_t sdcclSocketTryRecv(struct sdcclSocket *sock, void *ptr,
                                   int size, int *closed, bool blocking) {
  int offset = 0;
  if (sock == NULL) {
    WARN("sdcclSocketTryRecv: pass NULL socket");
    return sdcclInvalidArgument;
  }
  *closed = 0;
  // Block until connection closes or nbytes received
  if (blocking) {
    while (offset < size) {
      SDCCLCHECK(socketProgressOpt(SDCCL_SOCKET_RECV, sock, ptr, size,
                                    &offset, 0, closed));
      if (*closed)
        return sdcclSuccess;
    }
  } else {
    SDCCLCHECK(socketProgressOpt(SDCCL_SOCKET_RECV, sock, ptr, size, &offset,
                                  0, closed));
    if (*closed)
      return sdcclSuccess;

    // If any bytes were received, block waiting for the rest
    if (offset > 0) {
      while (offset < size) {
        SDCCLCHECK(socketProgressOpt(SDCCL_SOCKET_RECV, sock, ptr, size,
                                      &offset, 0, closed));
        if (*closed)
          return sdcclSuccess;
      }
      // No bytes were received, return sdcclInProgress
    } else {
      return sdcclInProgress;
    }
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketClose(struct sdcclSocket *sock) {
  if (sock != NULL) {
    if (sock->fd >= 0) {
      /* shutdown() is needed to send FIN packet to proxy thread; shutdown() is
       * not affected by refcount of fd, but close() is. close() won't close a
       * fd and send FIN packet if the fd is duplicated (e.g. fork()). So
       * shutdown() guarantees the correct and graceful connection close here.
       */
      shutdown(sock->fd, SHUT_RDWR);
      close(sock->fd);
    }
    sock->state = sdcclSocketStateClosed;
    sock->fd = -1;
  }
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketGetFd(struct sdcclSocket *sock, int *fd) {
  if (sock == NULL) {
    WARN("sdcclSocketGetFd: pass NULL socket");
    return sdcclInvalidArgument;
  }
  if (fd)
    *fd = sock->fd;
  return sdcclSuccess;
}

sdcclResult_t sdcclSocketSetFd(int fd, struct sdcclSocket *sock) {
  if (sock == NULL) {
    WARN("sdcclSocketGetFd: pass NULL socket");
    return sdcclInvalidArgument;
  }
  sock->fd = fd;
  return sdcclSuccess;
}
