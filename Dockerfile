FROM nvcr.io/nvidia/pytorch:25.03-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Create a non-root user
ENV USER=dev
ARG UID=1000
ARG GID=1000

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN userdel ubuntu || true
RUN groupadd --gid $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends sudo

RUN echo "$USER ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER \
    && chmod 0440 /etc/sudoers.d/$USER

USER $USER
WORKDIR /home/$USER

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
        build-essential cmake git ffmpeg ca-certificates curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH=/home/$USER/.local/bin:$PATH
CMD ["/bin/bash"]
