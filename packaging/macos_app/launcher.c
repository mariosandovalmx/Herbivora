#include <mach-o/dyld.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/*
 * Tiny universal entry point for HerbivoR.app.
 *
 * A shell script used directly as CFBundleExecutable makes Finder treat the
 * bundle as a script-only/generic app and can make Apple-silicon macOS launch
 * it through Rosetta. This native executable gives Launch Services a normal
 * universal app binary, then delegates to the maintained shell launcher.
 */
int main(void) {
    char executable[PATH_MAX];
    uint32_t size = sizeof(executable);
    if (_NSGetExecutablePath(executable, &size) != 0) {
        fputs("HerbivoR: executable path is too long\n", stderr);
        return 1;
    }

    char *slash = strrchr(executable, '/'); /* .../MacOS/HerbivoR */
    if (slash == NULL) {
        return 1;
    }
    *slash = '\0';
    slash = strrchr(executable, '/');       /* .../Contents/MacOS */
    if (slash == NULL) {
        return 1;
    }
    *slash = '\0';                         /* .../Contents */

    char launcher[PATH_MAX];
    int written = snprintf(
        launcher,
        sizeof(launcher),
        "%s/Resources/launcher.sh",
        executable
    );
    if (written < 0 || (size_t)written >= sizeof(launcher)) {
        fputs("HerbivoR: launcher path is too long\n", stderr);
        return 1;
    }

    execl("/bin/bash", "bash", launcher, (char *)NULL);
    perror("HerbivoR: could not start launcher");
    return 1;
}
