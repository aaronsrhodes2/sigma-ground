# The SkippyCraft client gametest: the real Minecraft client under a virtual display with software OpenGL, so a scripted
# player can place matter blocks and take screenshots with nobody at the keyboard (mod/test.sh).
FROM eclipse-temurin:25-jdk
RUN apt-get update && apt-get install -y --no-install-recommends xvfb xauth libgl1 libglx-mesa0 libgl1-mesa-dri mesa-vulkan-drivers \
    libxrandr2 libxxf86vm1 libxcursor1 libxi6 libxext6 libxrender1 libxtst6 libopenal1 libpulse0 x11-utils \
    && rm -rf /var/lib/apt/lists/*
ENV LIBGL_ALWAYS_SOFTWARE=1 MESA_GL_VERSION_OVERRIDE=4.5 MESA_GLSL_VERSION_OVERRIDE=450
